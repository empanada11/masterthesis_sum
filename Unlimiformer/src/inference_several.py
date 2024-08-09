import torch
from transformers import BartForConditionalGeneration, AutoTokenizer
from datasets import load_dataset
from rouge_score import rouge_scorer
import wandb
import argparse

# Argument parser for terminal execution
parser = argparse.ArgumentParser(description="Run Unlimiformer on a dataset and evaluate ROUGE scores")
parser.add_argument("--modelname", type=str, default="abertsch/unlimiformer-bart-govreport-alternating", help="Pretrained model name")
parser.add_argument("--dataset", type=str, default="urialon/gov_report_validation", help="HuggingFace dataset name")
parser.add_argument("--num_docs", type=int, default=50, help="Number of documents to process")
parser.add_argument("--wandb_project", type=str, default="unlimiformer-evaluation", help="wandb project name")
args = parser.parse_args()

# Initialize wandb
wandb.init(project=args.wandb_project)

# Set the device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained(args.modelname)
model.to(device)

# Load the dataset
dataset = load_dataset(args.dataset)
num_docs = min(args.num_docs, len(dataset['validation']))

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Prepare UnlimiformerArguments
defaults = UnlimiformerArguments()
unlimiformer_kwargs = {
    'layer_begin': defaults.layer_begin, 
    'layer_end': defaults.layer_end,
    'unlimiformer_head_num': defaults.unlimiformer_head_num, 
    'exclude_attention': defaults.unlimiformer_exclude, 
    'chunk_overlap': defaults.unlimiformer_chunk_overlap,
    'model_encoder_max_len': defaults.unlimiformer_chunk_size,
    'verbose': defaults.unlimiformer_verbose, 'tokenizer': tokenizer,
    'unlimiformer_training': defaults.unlimiformer_training,
    'use_datastore': defaults.use_datastore,
    'flat_index': defaults.flat_index,
    'test_datastore': defaults.test_datastore,
    'reconstruct_embeddings': defaults.reconstruct_embeddings,
    'gpu_datastore': defaults.gpu_datastore,
    'gpu_index': defaults.gpu_index
}

# Convert model using Unlimiformer
model = Unlimiformer.convert_model(model, **unlimiformer_kwargs)
model.eval()

# Process and evaluate documents
rouge_scores = []
for i in range(num_docs):
    example_input = dataset['validation'][i]['input']
    reference_summary = dataset['validation'][i]['output']  # Adjust according to your dataset structure
    
    # Tokenize input
    example = tokenizer(example_input, truncation=True, return_tensors="pt").to(device)
    
    # Generate output
    generated_output = tokenizer.batch_decode(model.generate(**example, max_length=512), ignore_special_tokens=True)[0]
    
    # Calculate ROUGE scores
    scores = scorer.score(reference_summary, generated_output)
    rouge_scores.append(scores)
    
    # Log the scores with wandb
    wandb.log({
        'document_index': i,
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    })

# Optionally print out average ROUGE scores
avg_rouge1 = sum([score['rouge1'].fmeasure for score in rouge_scores]) / num_docs
avg_rouge2 = sum([score['rouge2'].fmeasure for score in rouge_scores]) / num_docs
avg_rougeL = sum([score['rougeL'].fmeasure for score in rouge_scores]) / num_docs

print(f"Average ROUGE-1: {avg_rouge1}")
print(f"Average ROUGE-2: {avg_rouge2}")
print(f"Average ROUGE-L: {avg_rougeL}")

# Finish wandb run
wandb.finish()
