import torch
from transformers import AutoTokenizer, LongT5ForConditionalGeneration
from datasets import load_dataset
from rouge_score import rouge_scorer
import pandas as pd

# Set the device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the LongT5 model and tokenizer
model_name = "google/long-t5-tglobal-base"  # or "google/long-t5-tglobal-large" for larger model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LongT5ForConditionalGeneration.from_pretrained(model_name).to(device)

# Load the dataset (replace with your dataset)
dataset = load_dataset("urialon/gov_report_validation")
documents = dataset['validation']['input']  # Assuming the column name is 'input'
references = dataset['validation']['output']  # Assuming the column name is 'output'

# Prepare to save the results
output_data = []

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Process each document
for i, doc in enumerate(documents):
    # Tokenize the input with a max_length of 16k tokens
    inputs = tokenizer(doc, return_tensors="pt", truncation=True, max_length=16384).to(device)

    # Generate the summary
    summary_ids = model.generate(inputs.input_ids, max_length=512)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    # Calculate ROUGE scores
    reference = references[i]
    rouge_scores = scorer.score(reference, summary)
    
    # Save the results
    output_data.append({
        'document': doc,
        'reference': reference,
        'generated_summary': summary,
        'rouge1': rouge_scores['rouge1'].fmeasure,
        'rouge2': rouge_scores['rouge2'].fmeasure,
        'rougeL': rouge_scores['rougeL'].fmeasure
    })

# Convert output data to DataFrame and save to Excel
output_df = pd.DataFrame(output_data)
output_df.to_excel('longt5_inference_results.xlsx', index=False)

# Print average ROUGE scores
avg_rouge1 = output_df['rouge1'].mean()
avg_rouge2 = output_df['rouge2'].mean()
avg_rougeL = output_df['rougeL'].mean()

print(f"Average ROUGE-1: {avg_rouge1}")
print(f"Average ROUGE-2: {avg_rouge2}")
print(f"Average ROUGE-L: {avg_rougeL}")

print("Inference and evaluation completed. Results saved to 'longt5_inference_results.xlsx'")
