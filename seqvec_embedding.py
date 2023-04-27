# Install bio_embeddings using the command: pip install bio-embeddings[all]

from bio_embeddings.embed.seqvec_embedder import SeqVecEmbedder
import numpy as np
import torch 

seq = 'MVTYDFGSDEMHD' # A protein sequence of length L

embedder = SeqVecEmbedder()
embedding = embedder.embed(seq)
protein_embd = torch.tensor(embedding).sum(dim=0) # Vector with shape [L x 1024]
print(protein_embd)
np_arr = protein_embd.cpu().detach().numpy()
print(np_arr)