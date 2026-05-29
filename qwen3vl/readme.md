# Qwen2.5/3-VL 的 LoRA微调/部分参数微调代码示例

## 1. 文件夹格式
```text
.
├── Datas
│   └── LaTeX_OCR
│       └── human_handwrite_print_format
├── Models
│   ├── Qwen2.5-VL-3B-Instruct
│   └── Qwen3-VL-2B-Instruct
└── qwen3vl
    ├── qwen3vl_lora
    └── qwen3vl_ppft
```
其中，

数据集采用 LaTeX_OCR/human_handwrite_print_format（我们为每个样本增加了输入文本"Transcribe the LaTeX of this image."）。这是一个 LaTeX 印刷体公式数据集，需要模型识别数据集中的公式图像，输出 LaTeX 格式的文本。更多细节详见 https://huggingface.co/datasets/linxy/LaTeX_OCR

模型采用 Qwen2.5-VL-3B-Instruct 或 Qwen3-VL-2B-Instruct

LoRA 微调代码在 qwen3vl_lora 中，配置好参数后运行 `train.py` 即可。类似，部分参数（多模态投影层）微调代码在 qwen3vl_ppft 中，配置好参数后运行 `train.py` 即可。
