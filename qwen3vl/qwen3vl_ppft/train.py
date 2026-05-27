import matplotlib.pyplot as plt
import os

from transformers import TrainingArguments, Trainer

from config import parse_args
from data import get_data, Qwen3VLDataCollator
from model import load_tokenizer_processor, load_model, get_ppft_model


def train(args, tokenizer, train_dataset, eval_dataset, model):
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        logging_first_step=args.logging_first_step,
        per_device_train_batch_size=args.per_device_train_batch_size,  # 每个卡的批量大小
        learning_rate=args.learning_rate,  # 学习率
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # 梯度累加频率
        num_train_epochs=args.num_train_epochs,  # 训练轮数
        eval_strategy=args.eval_strategy,  # 模型测试集损失策略
        save_strategy=args.save_strategy,  # 模型保存策略
        save_total_limit=args.save_total_limit,  # 模型保存的最大数量
        gradient_checkpointing=True,  # 梯度检查点
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
    )
    trainer = Trainer(
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=Qwen3VLDataCollator(tokenizer=tokenizer),
        model=model,
    )
    trainer.train()

    logs = trainer.state.log_history
    steps = [log['step'] for log in logs if 'loss' in log]
    losses = [log['loss'] for log in logs if 'loss' in log]
    plt.plot(steps, losses)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(os.path.join(args.output_dir, "training_loss.png"))


def main(args):
    tokenizer, processor = load_tokenizer_processor(args)
    train_dataset, eval_dataset = get_data(args, tokenizer, processor)

    model = load_model(args)
    ppft_model = get_ppft_model(args, model)

    train(args, tokenizer, train_dataset, eval_dataset, ppft_model)


if __name__ == "__main__":
    args = parse_args()
    main(args)


