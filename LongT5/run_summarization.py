import logging
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    LongT5ForConditionalGeneration,
    LongT5Tokenizer,
    set_seed,
)
from datasets import load_dataset

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust remote code"}
    )

@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

@dataclass
class CustomTrainingArguments(TrainingArguments):
    gradient_checkpointing: bool = field(default=False)
    bf16: bool = field(default=False)


def chunk_text(text, max_length):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_length):
        chunks.append(" ".join(words[i:i + max_length]))
    return chunks


def preprocess_function(examples, tokenizer, max_source_length, max_target_length):
    inputs = examples["article"]
    targets = examples["abstract"]
    
    input_chunks = [chunk_text(text, max_source_length) for text in inputs]
    flattened_inputs = [chunk for sublist in input_chunks for chunk in sublist]
    flattened_targets = [target for target_list in [[target] * len(chunks) for target, chunks in zip(targets, input_chunks)] for target in target_list]

    model_inputs = tokenizer(
        flattened_inputs, max_length=max_source_length, truncation=True, return_tensors="pt", padding="max_length"
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            flattened_targets, max_length=max_target_length, truncation=True, return_tensors="pt", padding="max_length"
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    tokenizer = LongT5Tokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        trust_remote_code=model_args.trust_remote_code
    )

    model = LongT5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code
    )

    datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
    column_names = datasets["train"].column_names

    train_dataset = datasets["train"].map(
        lambda examples: preprocess_function(examples, tokenizer, data_args.max_source_length, data_args.max_target_length),
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on train dataset",
    )

    eval_dataset = datasets["validation"].map(
        lambda examples: preprocess_function(examples, tokenizer, data_args.max_source_length, data_args.max_target_length),
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on validation dataset",
    )

    training_args.remove_unused_columns = False
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        predict_results = trainer.predict(test_dataset=eval_dataset)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)


if __name__ == "__main__":
    main()
