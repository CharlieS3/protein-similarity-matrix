import os
import re
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from transformers import T5Model, T5Tokenizer, T5EncoderModel, AutoTokenizer, AutoModelForSeq2SeqLM

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cos = torch.nn.CosineSimilarity(dim=0)

def ProtT5_initialize():
    #load the model and tokenizer
    ProtT5 = T5EncoderModel.from_pretrained("protT5", local_files_only=True)
    ProtT5_tokenizer = T5Tokenizer.from_pretrained("protT5", do_lower_case=False)

    ProtT5 = ProtT5.half()
    ProtT5 = ProtT5.eval()
    ProtT5 = ProtT5.to(device)

    return ProtT5, ProtT5_tokenizer

def get_embs_T5(seq, T5, ProtT5_tokenizer):

  seq = re.sub(r"[UZOB]", "X", seq)
  seq = " ".join(list(seq))

  inputs = ProtT5_tokenizer([seq], padding=True, return_tensors="pt") #pytorch format 
  inputs = {key: value.to(device) for key, value in inputs.items()}

  #get the embeddings (last hidden state)
  with torch.no_grad(): #no gradients since we are not training
    outputs = T5(**inputs) #send inputs through the model
    last_hidden_states = outputs.last_hidden_state #get an embedding for each token
  
  sequence_embedding = last_hidden_states[0, :-1, :]
  return sequence_embedding

def compute_similarity_matrix(seq_1, seq_2, emb1, emb2):

  len1, len2 = len(seq_1), len(seq_2)
  similarity_matrix = np.zeros((len1, len2))

  for i in range(len1):
      for j in range(len2):
          similarity_matrix[i, j] = cos(torch.tensor(emb1[i] , dtype = torch.float32), torch.tensor(emb2[j] , dtype = torch.float32)).item()

  return similarity_matrix

def plot_heatmap(similarity_matrix, seq1, seq2, file_path):
  plt.figure(figsize=(10, 8))
  sns.heatmap(similarity_matrix, xticklabels=list(seq2), yticklabels=list(seq1), cmap="coolwarm")
  plt.xlabel('Sequence 2')
  plt.ylabel('Sequence 1')
  plt.title('Cosine Similarity Heatmap')

  # Save the heatmap to the specified file path
  plt.savefig(file_path, format='png', bbox_inches='tight')
  plt.close()  # Free memory

def prepare_training_data(data_df, model, tokenizer, output_dir="output_files"):

    #ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    similarity_matrices = []
    region_indices = []

    # Process a subset for debugging (remove this limit for full training)
    for i, row in data_df.iterrows():
        try:
            #extract sequences
            seq1 = row['protein1']
            seq2 = row['protein2']
            # Get embeddings
            emb1 = get_embs_T5(seq1, model, tokenizer).cpu().numpy()
            emb2 = get_embs_T5(seq2, model, tokenizer).cpu().numpy()

            # Get the example
            similarity_matrix = compute_similarity_matrix(seq1, seq2, emb1, emb2)

            #extract region indices
            i1, j1 = int(row['p1_mStartIndex']), int(row['p2_mStartIndex'])
            i4, j4 = int(row['p1_mEndIndex']), int(row['p2_mEndIndex'])
            i2, j2 = i1, j4  # Top-right
            i3, j3 = i4, j1  # Bottom-left

            # Storing in lists
            similarity_matrices.append(similarity_matrix)
            region_indices.append([(i1, j1), (i2, j2), (i3, j3), (i4, j4)])

            # Dsave progress after every 10 examples
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1} examples")
        except Exception as e:
            print(f"Error processing row {i}: {e}")
            continue

    #save similarity matrices
    # np.save(os.path.join(output_dir, "similarity_matrices.npy"), np.array(similarity_matrices))
    np.save(os.path.join(output_dir, "similarity_matrices.npy"), np.array(similarity_matrices, dtype=object))


    # Save region indices as a text file with each list on a new line
    with open(os.path.join(output_dir, "label_indices.txt"), 'w') as f:
        for region in region_indices:
            f.write(f"{region}\n")
    
    return similarity_matrices, region_indices

def main():
    data_df = pd.read_csv("merged_CDD.csv")
    output_dir="output_files_CDD"
    model, tokenizer = ProtT5_initialize()
    similarity_matrices, region_indices = prepare_training_data(data_df, model, tokenizer, output_dir)


if __name__ == "__main__":
    main()