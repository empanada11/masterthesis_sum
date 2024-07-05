import sys
import os
import warnings
import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from transformers import DataCollatorForSeq2Seq
from accelerate import Accelerator

def main():
    parser = argparse.ArgumentParser()

    # Adding the arguments
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--max_source_length", type=int, default=8192)
    parser.add_argument("--max_target_length", type=int, default=1024)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=12)
    parser.add_argument("--optim", type=str, default="adafactor")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--lr_scheduler_type", type=str, default="constant")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--predict_with_generate", action="store_true")
    parser.add_argument("--generation_num_beams", type=int, default=1)
    parser.add_argument("--generation_max_length", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--report_to", type=str, default="all")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=5)
    parser.add_argument("--evaluation_strategy", type=str, default="steps")
    parser.add_argument("--ddp_find_unused_parameters", action="store_false", dest="find_unused_parameters")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")

    args = parser.parse_args()

    # Initialize Accelerator for multi-GPU support
    accelerator = Accelerator()

    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=FutureWarning, message="The `device` argument is deprecated")
    warnings.filterwarnings("ignore", category=UserWarning, message="torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly")

    # Load the LongT5 model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # For gradient checkpointing
    model.config.use_cache = False
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=4,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q", "v"]
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Enable gradient computation for encoder and decoder
    model.base_model.model.encoder.enable_input_require_grads()
    model.base_model.model.decoder.enable_input_require_grads()

    dataset = load_dataset(args.dataset_name)
    train_dataset = dataset['train'].shuffle(seed=42).select(range(100))  # Use only 100 samples for training
    eval_dataset = dataset['validation'].shuffle(seed=42).select(range(10))  # Use only 10 samples for evaluation

    def preprocess_function(examples):
        inputs = [ex for ex in examples['document']]
        targets = [ex for ex in examples['summary']]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, truncation=True)
        labels = tokenizer(targets, max_length=args.max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = train_dataset.map(preprocess_function, batched=True)
    eval_dataset = eval_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=args.num_train_epochs,
        predict_with_generate=args.predict_with_generate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        use_reentrant=False,  # Explicitly set use_reentrant to avoid warning
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=500,
        load_best_model_at_end=True,
        report_to=args.report_to,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        run_name=args.run_name,
        ddp_find_unused_parameters=args.find_unused_parameters,
        no_cuda=args.no_cuda,
        bf16=args.bf16
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )

    # Prepare everything with accelerator
    trainer.accelerator = accelerator

    # Start training
    if args.do_train:
        trainer.train()
    if args.do_eval:
        trainer.evaluate()
    if args.do_predict:
        trainer.predict(test_dataset=eval_dataset)

if __name__ == "__main__":
    main()
