import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # 实验配置
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="随机种子")
    parser.add_argument("--output_dir",
                        type=str,
                        default="outputs",
                        help="输出文件夹路径")
    #数据
    parser.add_argument("--data_dir",
                        type=str,
                        default="../../Datas/LaTeX_OCR/human_handwrite_print_format",
                        help="数据文件夹路径")
    parser.add_argument("--train_fraction",
                        type=float,
                        default=0.5,
                        help="训练数据抽样比例")
    parser.add_argument("--test_fraction",
                        type=float,
                        default=1.0,
                        help="测试数据抽样比例")
    parser.add_argument("--if_default_text",
                        type=bool,
                        default=True,
                        help="是否为每一个message都使用默认输入文本")
    parser.add_argument("--default_text",
                        type=str,
                        default="Transcribe the LaTeX of this image.",
                        help="默认输入文本")
    parser.add_argument("--max_length",
                        type=int,
                        default=8192,
                        help="输入序列的最大长度")
    #模型
    parser.add_argument("--model_dir",
                        type=str,
                        # default="../../Models/Qwen2.5-VL-3B-Instruct",
                        default="../../Models/Qwen3-VL-2B-Instruct",
                        help="预训练模型文件夹路径")
    #模型_ppft
    parser.add_argument("--trainable_keywords",
                        type=str,
                        default=["visual.merger"],
                        help="可训练参数关键词列表",
                        nargs='+')
    #训练
    parser.add_argument("--logging_steps",
                        type=int,
                        default=20,
                        help="日志打印频率")
    parser.add_argument("--logging_first_step",
                        type=bool,
                        default=True,
                        help="是否在第一步打印日志")
    parser.add_argument("--per_device_train_batch_size",
                        type=int,
                        default=8,
                        help="每个卡的批量大小")
    parser.add_argument("--learning_rate",
                        type=float,
                        default=1e-4,
                        help="学习率")
    parser.add_argument("--gradient_accumulation_steps",
                        type=int,
                        default=1,
                        help="梯度累加频率")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=5,
                        help="训练轮数")
    parser.add_argument("--eval_strategy",
                        type=str,
                        default='epoch',
                        help="模型测试集损失策略。可选'no'(不测试),'epoch'(每个epoch结束时测试),'steps'(每隔save_steps步数测试)")
    parser.add_argument("--save_strategy",
                        type=str,
                        default='no',
                        help="模型保存策略。可选'no'(不保存),'epoch'(每个epoch结束时保存),'steps'(每隔save_steps步数保存)")
    parser.add_argument("--save_total_limit",
                        type=int,
                        default=0,
                        help="模型保存的最大数量")

    args = parser.parse_args()
    return args