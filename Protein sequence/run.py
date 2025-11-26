import os
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd

# Set environment variable to disable HuggingFace Hub's symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Set device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Specify local models path
model_local_path = r"D:\Protein\facebook_esm2_t6_8M_UR50D\models--facebook--esm2_t6_8M_UR50D\snapshots\c731040fcd8d73dceaa04b0a8e6329b345b0f5df"

# Load local ESM models and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_local_path)
model = AutoModel.from_pretrained(model_local_path, output_hidden_states=True)
model.to(device)
model.eval()

# Read data file
data_file = r"D:\Protein\test.csv"
df = pd.read_csv(data_file)

# Extract necessary information
drugbank_ids = df["DrugBank_ID"].tolist()
uniprot_ids = df["UniProt_ID"].tolist()
fasta_sequences = df["FASTA_Sequence"].tolist()

print(f"There are {len(df)} protein sequences to process.")

# Initialize embedding list, pre-filled with None
sequence_embeddings = [None] * len(df)

# Determine the dimension of the embedding vector
# Use a valid sample to get the embedding dimension
for seq in fasta_sequences:
    if pd.notna(seq):
        encoded_input = tokenizer(seq, return_tensors='pt', padding=True)
        with torch.no_grad():
            outputs = model(
                encoded_input['input_ids'].to(device),
                attention_mask=encoded_input['attention_mask'].to(device),
            )
        hidden_states = outputs.hidden_states
        last_hidden_states = hidden_states[-1]
        embedding_size = last_hidden_states.size(-1)
        break
else:
    raise ValueError("All sequences are empty, unable to determine the embedding vector dimension.")

# Create a zero vector
zero_vector = np.zeros(embedding_size, dtype=np.float32)

# Set batch size
batch_size = 4  # Adjust according to GPU memory

# Record the number of skipped samples (optional)
skipped_count = 0

for i in range(0, len(fasta_sequences), batch_size):
    batch_indices = list(range(i, min(i + batch_size, len(fasta_sequences))))
    batch_drugbank_ids = [drugbank_ids[idx] for idx in batch_indices]
    batch_uniprot_ids = [uniprot_ids[idx] for idx in batch_indices]
    batch_sequences = [fasta_sequences[idx] for idx in batch_indices]

    # Identify valid samples (DrugBank_ID, UniProt_ID, and FASTA_Sequence are all non-NULL)
    valid_indices = [j for j, (drug, uni, seq) in enumerate(zip(batch_drugbank_ids, batch_uniprot_ids, batch_sequences))
                     if pd.notna(drug) and pd.notna(uni) and pd.notna(seq)]

    if not valid_indices:
        # If the entire batch has no valid sequences, fill all corresponding positions with zero vectors
        for j in batch_indices:
            sequence_embeddings[j] = zero_vector
        skipped_count += len(batch_sequences)
        continue

    # Extract sequences and original indices of valid samples
    valid_batch_sequences = [batch_sequences[j] for j in valid_indices]
    valid_batch_indices = [batch_indices[j] for j in valid_indices]

    # Encode valid sequences
    encoded_inputs = tokenizer(valid_batch_sequences, return_tensors='pt', padding=True)
    batch_tokens = encoded_inputs['input_ids'].to(device)
    batch_attention_mask = encoded_inputs['attention_mask'].to(device)
    batch_lens = batch_attention_mask.sum(dim=1)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs = model(
                batch_tokens,
                attention_mask=batch_attention_mask,
            )
            hidden_states = outputs.hidden_states
            last_hidden_states = hidden_states[-1]

    # Generate embedding features for each sequence (via mean pooling)
    for j, tokens_len in enumerate(batch_lens):
        valid_token_reps = last_hidden_states[j, 1: tokens_len - 1]  # Exclude start and end special tokens
        sequence_rep = valid_token_reps.mean(dim=0)  # (hidden_size,)
        sequence_embeddings[valid_batch_indices[j]] = sequence_rep.cpu().numpy()

    # For invalid samples, fill with zero vectors
    invalid_in_batch = set(batch_indices) - set(valid_batch_indices)
    for j in invalid_in_batch:
        sequence_embeddings[j] = zero_vector

    # Clear cache
    torch.cuda.empty_cache()

# Fill all unassigned embeddings with zero vectors (just in case)
for idx in range(len(sequence_embeddings)):
    if sequence_embeddings[idx] is None:
        sequence_embeddings[idx] = zero_vector

# Create a DataFrame of embedding features
embedding_df = pd.DataFrame({
    "DrugBank_ID": drugbank_ids,
    "UniProt_ID": uniprot_ids,
    "Embedding": sequence_embeddings
})

# Save embedding features to file
embedding_dir = r"D:\Protein\embeddings3"
os.makedirs(embedding_dir, exist_ok=True)
embedding_file = os.path.join(embedding_dir, "protein_embeddings.npy")
np.save(embedding_file, embedding_df["Embedding"].tolist())
print(f"All protein embeddings have been saved to {embedding_file}")

# Optional: Save as CSV file (embeddings stored as strings)
csv_embedding_file = os.path.join(embedding_dir, "protein_embeddings.csv")
embedding_df.to_csv(csv_embedding_file, index=False)
print(f"All protein embeddings have been saved to {csv_embedding_file}")

# Optional: Print the shape of the embedding features
for idx, embedding in enumerate(embedding_df["Embedding"]):
    print(f"Embedding shape for {embedding_df['DrugBank_ID'][idx]} ({embedding_df['UniProt_ID'][idx]}): {embedding.shape}")