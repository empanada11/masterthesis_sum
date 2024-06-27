import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    set_seed,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from datasets import load_dataset, load_metric

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"}
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
            "help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "The number of processes to use for the preprocessing."}
    )

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.setLevel(logging.INFO)

    set_seed(training_args.seed)

    # Load the dataset
    raw_datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)

    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)

    # Preprocessing the datasets
    def preprocess_function(examples):
        inputs = examples["article"]
        targets = examples["abstract"]
        model_inputs = tokenizer(
            inputs, max_length=data_args.max_source_length, truncation=True, padding="max_length"
        )
        labels = tokenizer(
            targets, max_length=data_args.max_target_length, truncation=True, padding="max_length"
        ).input_ids
        model_inputs["labels"] = labels
        return model_inputs

    # Apply chunking to the dataset
    def chunk_examples(examples):
        chunks = []
        for example in examples["article"]:
            chunked_texts = [example[i:i + data_args.max_source_length] for i in range(0, len(example), data_args.max_source_length)]
            chunks.extend(chunked_texts)
        return {"article": chunks, "abstract": examples["abstract"] * len(chunked_texts)}

    # Map chunking function to the dataset
    raw_datasets = raw_datasets.map(chunk_examples, batched=True)
    
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True, num_proc=data_args.preprocessing_num_workers)

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    # Initialize the Seq2SeqTrainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # Training
    if training_args.do_train:
        trainer.train()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        test_results = trainer.predict(test_dataset=tokenized_datasets["test"])
        trainer.log_metrics("test", test_results.metrics)
        trainer.save_metrics("test", test_results.metrics)

if __name__ == "__main__":
    main()
