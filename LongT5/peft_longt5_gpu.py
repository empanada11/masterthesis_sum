import sys
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback, EarlyStoppingCallback
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from transformers import DataCollatorForSeq2Seq
from accelerate import Accelerator

def main():
    # Initialize Accelerator for multi-GPU support
    accelerator = Accelerator()

    # Load the LongT5 model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained("google/long-t5-local-base")
    tokenizer = AutoTokenizer.from_pretrained("google/long-t5-local-base")

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

    dataset = load_dataset("mimiklee/masterthesis-longt5-sum-8k")
    train_dataset = dataset['train']
    eval_dataset = dataset['validation']

    def preprocess_function(examples):
        inputs = [ex for ex in examples['document']]
        targets = [ex for ex in examples['summary']]
        model_inputs = tokenizer(inputs, max_length=8192, truncation=True)
        labels = tokenizer(targets, max_length=1024, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = train_dataset.map(preprocess_function, batched=True)
    eval_dataset = eval_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",  # Changed to epoch to match save_strategy
        save_strategy="epoch",  # Ensuring both strategies are the same
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        lr_scheduler_type='linear',
        warmup_steps=500,
        load_best_model_at_end=True,
        report_to="none"  # Avoid reporting to external services
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]  # Early stopping after 1 epoch of no improvement
    )

    # Prepare everything with accelerator
    trainer.accelerator = accelerator

    trainer.train()

if __name__ == "__main__":
    main()
