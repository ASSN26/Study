import os
from typing import Any, Dict, List

from datasets import load_dataset
from qwen_vl_utils import process_vision_info
import torch


def process_func(example, args, tokenizer, processor):
    # 从example中提取图像、文本、目标文本
    image = example["image"]
    if args.if_default_text:
        input_content = args.default_text
    else:
        input_content = example["text"]["user"]
    output_content = example["text"]["assistant"]
    # 构建qwen2.5vl/3vl的结构化消息
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": input_content},
            ],
        }
    ]
    # 将结构化消息转换成特定的纯文本字符串。其中，将图像替换为占位符<|vision_start|><|image_pad|><|vision_end|>。
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # 从结构化消息中提取视觉数据
    image_inputs, video_inputs = process_vision_info(messages)
    # 将纯文本字符串、视觉数据转换成Tensor。其中，将文本转换成Token_ID，将图像缩放、裁剪、归一化……转换成Tensor。
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        do_resize=True,
    )
    instruction_input_ids = inputs["input_ids"][0]
    instruction_attention_mask = inputs["attention_mask"][0]
    instruction_pixel_values = inputs["pixel_values"]
    instruction_image_grid_thw = inputs["image_grid_thw"][0]
    instruction_mm_token_type_ids = inputs["mm_token_type_ids"][0]

    # 单独将目标文本转换成Tensor
    response = tokenizer(f"{output_content}", add_special_tokens=False)
    response_input_ids = response["input_ids"]
    response_attention_mask = response.get("attention_mask", [1]*len(response_input_ids))
    response_mm_token_type_ids = [0] * len(response_input_ids)
    # 添加结束符。
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None:
        if not response_input_ids or response_input_ids[-1] != eos_token_id:
            response_input_ids = response_input_ids + [eos_token_id]
            response_attention_mask = response_attention_mask + [1]
            response_mm_token_type_ids = response_mm_token_type_ids + [0]
    else:
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("需要定义 eos_token_id 或 pad_token_id 才能结束响应序列。")
        response_input_ids = response_input_ids + [pad_token_id]
        response_attention_mask = response_attention_mask + [1]
        response_mm_token_type_ids = response_mm_token_type_ids + [0]
    # 将用户输入和目标文本拼接在一起，形成一条完整的对话序列。对于标签，用[-100]把用户输入掩码，使模型不学习预测这部分。
    input_ids = instruction_input_ids + response_input_ids
    attention_mask = instruction_attention_mask + response_attention_mask
    labels = ([-100] * len(instruction_input_ids) + response_input_ids)
    mm_token_type_ids = instruction_mm_token_type_ids + response_mm_token_type_ids
    if len(input_ids) > args.max_length:
        input_ids = input_ids[:args.max_length]
        attention_mask = attention_mask[:args.max_length]
        labels = labels[:args.max_length]
        mm_token_type_ids = mm_token_type_ids[:args.max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": instruction_pixel_values,
        "image_grid_thw": instruction_image_grid_thw,
        "mm_token_type_ids": mm_token_type_ids,
    }


class Qwen3VLDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_id_tensors = [torch.as_tensor(sample["input_ids"], dtype=torch.long) for sample in features]
        attention_tensors = [torch.as_tensor(sample["attention_mask"], dtype=torch.long) for sample in features]
        label_tensors = [torch.as_tensor(sample["labels"], dtype=torch.long) for sample in features]
        mm_token_type_tensors = [torch.as_tensor(sample["mm_token_type_ids"], dtype=torch.long) for sample in features]

        max_length = max(t.size(0) for t in input_id_tensors)
        pad_id = (
            self.tokenizer.pad_token_id
            if getattr(self.tokenizer, "pad_token_id", None) is not None
            else self.tokenizer.eos_token_id
        )
        if pad_id is None:
            raise ValueError("pad_token_id 与 eos_token_id 均为 None，无法进行padding。")

        input_ids = torch.full((len(features), max_length), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(features), max_length), dtype=torch.long)
        labels = torch.full((len(features), max_length), -100, dtype=torch.long)
        mm_token_type_ids = torch.zeros((len(features), max_length), dtype=torch.long)
        for idx, (ids, attn, lbl, mm_ids) in enumerate(zip(input_id_tensors, attention_tensors, label_tensors, mm_token_type_tensors)):
            length = ids.size(0)
            input_ids[idx, :length] = ids
            attention_mask[idx, :length] = attn
            labels[idx, :length] = lbl
            mm_token_type_ids[idx, :length] = mm_ids

        pixel_tensors = []
        for sample in features:
            pv = sample["pixel_values"]
            if not isinstance(pv, torch.Tensor):
                pv = torch.tensor(pv, dtype=torch.float32)
            pixel_tensors.append(pv)
        pixel_values = torch.cat(pixel_tensors, dim=0)

        image_grid_thw = torch.stack([torch.as_tensor(sample["image_grid_thw"], dtype=torch.long).view(-1) for sample in features], dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "mm_token_type_ids": mm_token_type_ids,
        }

def get_data(args, tokenizer, processor):
    ds = load_dataset("parquet", data_files={"train": os.path.join(args.data_dir, "train-*.parquet"),
                                                  "test": os.path.join(args.data_dir, "test-*.parquet"),})
    ds = ds.shuffle(seed=args.seed)
    train_data = ds["train"].select(range(int(len(ds["train"]) * args.train_fraction)))
    print(f"训练数据大小: {len(train_data)}")
    test_data = ds["test"].select(range(int(len(ds["test"]) * args.test_fraction)))
    print(f"测试数据大小: {len(test_data)}")

    map_kwargs = {"args":args, "tokenizer": tokenizer, "processor": processor}
    train_dataset = train_data.map(process_func,
                                   remove_columns=train_data.column_names,
                                   fn_kwargs=map_kwargs)
    eval_dataset = test_data.map(process_func,
                                 remove_columns=test_data.column_names,
                                 fn_kwargs=map_kwargs,)
    return train_dataset, eval_dataset