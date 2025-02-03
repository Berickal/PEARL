import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

model_name = "pythia_410m_ft_40"  # Adjust this to your specific Pythia model
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Load the CSV file
csv_file = "eval_data_1k.csv"
df = pd.read_csv(csv_file)

# Batch processing
batch_size = 24
prompts = df["prompt"].tolist()

# Add new columns for each response iteration
for iteration in range(0, 8):  # Loop for 8 iterations
    results = []
    
    print(f"Starting iteration {iteration}...")
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        # Generate responses
        outputs = model.generate(**inputs, max_new_tokens=100)  # Adjust max_new_tokens as needed
        responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Store results temporarily
        results.extend(responses)
    
    # Add results to the DataFrame
    df[f"response_{iteration}"] = results
    print(f"Iteration {iteration} completed.")

# Save the results to a CSV file
output_file = "output_check.40.csv"
df.to_csv(output_file, index=False)

print(f"All iterations completed. Responses saved to '{output_file}'.")

