def print_model_structure(model, max_depth=3):
    """打印模型结构，帮助确定要训练的层"""
    print("\n模型结构:")
    for name, module in model.named_modules():
        depth = name.count('.')
        if depth <= max_depth:
            indent = "  " * depth
            print(f"{indent}{name}: {module.__class__.__name__}")