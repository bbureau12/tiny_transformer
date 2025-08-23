# embedding_demo.py
# Minimal demo: embed a phrase, then extend it and show embeddings.

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Simple vocabulary ---
vocab = ["<PAD>", "<BOS>", "<EOS>", "the", "cat", "sat", "on", "mat", "dog", "ran"]
stoi = {w: i for i, w in enumerate(vocab)}
itos = {i: w for w, i in stoi.items()}

# --- Embedding layer ---
embedding_dim = 8  # keep small for display
emb = nn.Embedding(len(vocab), embedding_dim)

def encode(text):
    return [stoi.get(tok, 0) for tok in text.lower().split()]

def show_embeddings(tokens):
    ids = torch.tensor(tokens)
    vecs = emb(ids).detach()
    for tok_id, vec in zip(tokens, vecs):
        print(f"{itos[tok_id]:>5} -> {vec.numpy()}")

def cosine_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

# --- Demo ---
if __name__ == "__main__":
    # Phrase 1
    phrase1 = "the cat sat"
    ids1 = encode(phrase1)
    print(f"\nEmbeddings for: '{phrase1}'")
    show_embeddings(ids1)

    # Phrase 2 (add words)
    phrase2 = "the cat sat on the mat"
    ids2 = encode(phrase2)
    print(f"\nEmbeddings for: '{phrase2}'")
    show_embeddings(ids2)

    # Show similarity example
    cat_vec = emb(torch.tensor(stoi["cat"])).detach()
    dog_vec = emb(torch.tensor(stoi["dog"])).detach()
    mat_vec = emb(torch.tensor(stoi["mat"])).detach()

    print("\nSimilarity check:")
    print(f"cosine(cat, dog) = {cosine_sim(cat_vec, dog_vec):.3f}")
    print(f"cosine(cat, mat) = {cosine_sim(cat_vec, mat_vec):.3f}")