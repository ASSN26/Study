import importlib

from peft import LoraConfig, TaskType, get_peft_model
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

def get_lora_model(args, model):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=args.target_modules,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    lora_model = get_peft_model(model, config)
    lora_model.enable_input_require_grads()
    return lora_model