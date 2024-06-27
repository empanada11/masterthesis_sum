import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import transformers
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    max_source_length: Optional[int] = field(
        default=8192,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated."
        },
    )
    max_target_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer than this will be truncated."
        },
    )

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

def chunk_text(text, chunk_size):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield ' '.join(words[i:i + chunk_size])

def preprocess_function(examples, tokenizer, chunk_size):
    inputs = examples["article"]
    targets = examples["abstract"]

    chunked_inputs = []
    chunked_targets = []

    for input_text, target_text in zip(inputs, targets):
        input_chunks = list(chunk_text(input_text, chunk_size))
        for chunk in input_chunks:
            chunked_inputs.append(chunk)
            chunked_targets.append(target_text)

    model_inputs = tokenizer(chunked_inputs, max_length=chunk_size, padding="max_length", truncation=True)
    labels = tokenizer(chunked_targets, max_length=512, padding="max_length", truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    )

    # Load dataset
    dataset = load_dataset(data_args.dataset_name)

    # Preprocess dataset with chunking
    tokenized_datasets = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, data_args.max_source_length),
        batched=True,
        remove_columns=["article", "abstract"],
    )

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"] if training_args.do_train else None,
        eval_dataset=tokenized_datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload
        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_length=data_args.max_target_length)
        max_eval_samples = data_args.max_eval_samples
        metrics["eval_samples"] = min(max_eval_samples, len(tokenized_datasets["validation"]))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        predict_results = trainer.predict(
            tokenized_datasets["test"], metric_key_prefix="predict", max_length=data_args.max_target_length
        )
        metrics = predict_results.metrics
        max_predict_samples = data_args.max_predict_samples
        metrics["predict_samples"] = min(max_predict_samples, len(tokenized_datasets["test"]))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

if __name__ == "__main__":
    main()
