import importlib

import torch
from transformers import AutoProcessor, AutoTokenizer, AutoConfig


def load_tokenizer_processor(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_dir, use_fast=False)
    return tokenizer, processor

def load_model(args):
    # 直接加载配置文件中指定的原始模型类，加载模型配置->从配置中提取模型架构的类名->导入该模型类->实例化模型
    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    arch = (config.architectures or [None])[0]
    module_name = f"transformers.models.{config.model_type}.modeling_{config.model_type}"
    module = importlib.import_module(module_name)
    model_cls = getattr(module, arch)
    model = model_cls.from_pretrained(
        args.model_dir,
        device_map="auto",
        trust_remote_code=True,
    )
    model.to(dtype=torch.bfloat16)
    model.config.use_cache = False
    return model


def get_ppft_model(args, model):
    """
    冻结模型，仅将包含指定关键词的参数设置为可训练。
    Args:
        model: 模型实例
        trainable_keywords: 关键词列表，例如：["visual.merger"]
    Returns:
        model: 设置完成后的模型
    """
    for param in model.parameters():
        param.requires_grad = False

    trainable_keywords = args.trainable_keywords
    matched_param_names = []
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in trainable_keywords):
            param.requires_grad = True
            matched_param_names.append(name)

    total_params = 0
    trainable_params = 0
    for param in model.parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params

    print("=" * 80)
    print("可训练参数关键词: ", trainable_keywords)
    print(f"匹配的可训练参数: {len(matched_param_names)}")
    for n in matched_param_names:
        print(f"[Trainable] {n}")
    print("-" * 80)
    print(f"可训练参数量: {trainable_params:,}")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数占比: {100 * trainable_params / total_params:.4f}%")
    print("=" * 80)

    if len(matched_param_names) == 0:
        raise ValueError(f"未找到任何与关键词 {trainable_keywords} 匹配的参数，请检查模型参数命名。")
    return model