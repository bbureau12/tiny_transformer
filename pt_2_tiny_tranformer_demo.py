# tiny_transformer_10.py
# Minimal next-token LM with a tiny Transformer on 10 fixed sentences + optional attention heatmap.

import math, random
from typing import List
import torch
import torch.nn as nn

# ---------- Toy corpus ----------
TEXTS = [
    "the cat sat on the mat",
    "the dog chased the ball",
    "birds fly in the sky",
    "robots learn from data",
    "the coder writes clean code",
    "a frog sings at night",
    "the child reads a book",
    "rain falls on the roof",
    "the teacher asks a question",
    "the friend makes tea"
]

# ---------- Tokenizer ----------
class WordTokenizer:
    def __init__(self, texts: List[str], specials=("<PAD>", "<BOS>", "<EOS>", "<UNK>")):
        vocab = set()
        for t in texts:
            vocab.update(t.lower().split())
        self.pad, self.bos, self.eos, self.unk = specials
        self.itos = [self.pad, self.bos, self.eos, self.unk] + sorted(vocab)
        self.stoi = {w:i for i,w in enumerate(self.itos)}
        self.pad_id = self.stoi[self.pad]
        self.bos_id = self.stoi[self.bos]
        self.eos_id = self.stoi[self.eos]
        self.unk_id = self.stoi[self.unk]

    def encode(self, text: str):
        ids = [self.bos_id] + [self.stoi.get(w, self.unk_id) for w in text.lower().split()] + [self.eos_id]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids):
        words = []
        for i in ids:
            if i in (self.pad_id, self.bos_id, self.eos_id):
                continue
            words.append(self.itos[i])
        return " ".join(words)

def make_pairs(texts, tok: WordTokenizer):
    X, Y = [], []
    for t in texts:
        ids = tok.encode(t)
        X.append(ids[:-1])  # input
        Y.append(ids[1:])   # target
    # simple pad to same length
    max_len = max(len(x) for x in X)
    def pad(seq): return torch.cat([seq, torch.full((max_len-len(seq),), tok.pad_id, dtype=torch.long)])
    X = torch.stack([pad(x) for x in X])  # (B,T)
    Y = torch.stack([pad(y) for y in Y])  # (B,T)
    pad_mask = (X == tok.pad_id)         # (B,T)
    return X, Y, pad_mask

# ---------- Positional encoding ----------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)  # even dims → sine
        pe[:, 1::2] = torch.cos(pos * div)  # odd  dims → cosine
        self.register_buffer("pe", pe)

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)

# ---------- Tiny Transformer with attention weights exposure ----------
class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2, dim_ff=128, pad_id=0, dropout=0.1):
        super().__init__()
        self.pad_id = pad_id
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = SinusoidalPositionalEncoding(d_model)

        # custom encoder layer so we can grab attention weights
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

        # hook storage
        self.last_attn = None
        # register forward hook on first layer's self_attn to capture attn weights
        def hook_attn(module, input, output):
            # PyTorch's MultiheadAttention returns (attn_output, attn_weights)
            # but inside TransformerEncoderLayer it's not directly exposed.
            # So we monkey-patch by replacing forward with one that returns weights.
            pass  # Kept simple; we’ll compute an external attention pass for demo instead.

    def causal_mask(self, T, device):
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, x, pad_mask):
        B, T = x.shape
        h = self.tok_emb(x)
        h = self.pos(h)
        mask = self.causal_mask(T, x.device)
        out = self.encoder(h, mask=mask, src_key_padding_mask=pad_mask)  # (B,T,D)
        logits = self.lm_head(out)
        return logits, out

    # lightweight attention demo for 1 head on the final hidden states (not the internal mha)
    def single_head_attention_demo(self, hidden, head_dim=16):
        """
        Compute a toy 1-head attention over 'hidden' just for visualization.
        hidden: (B,T,D). We make Q,K,V by linear projections and return attn weights for B=1.
        """
        B, T, D = hidden.shape
        assert B == 1, "demo assumes batch size 1"
        Wq = nn.Linear(D, head_dim, bias=False).to(hidden.device)
        Wk = nn.Linear(D, head_dim, bias=False).to(hidden.device)
        Wv = nn.Linear(D, head_dim, bias=False).to(hidden.device)
        with torch.no_grad():
            Q = Wq(hidden)  # (1,T,Hd)
            K = Wk(hidden)
            scores = (Q @ K.transpose(1,2)) / math.sqrt(head_dim)  # (1,T,T)
            # causal mask for demo
            causal = torch.triu(torch.ones(T, T, device=hidden.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal, float('-inf'))
            attn = torch.softmax(scores, dim=-1)  # (1,T,T)
        return attn[0]  # (T,T)

# ---------- Train / sample ----------
def train(epochs=40, lr=3e-3, d_model=64, nhead=4, num_layers=2, dim_ff=128, seed=7, device=None):
    random.seed(seed); torch.manual_seed(seed)
    tok = WordTokenizer(TEXTS)
    X, Y, pad_mask = make_pairs(TEXTS, tok)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X, Y, pad_mask = X.to(device), Y.to(device), pad_mask.to(device)

    model = TinyTransformerLM(len(tok.itos), d_model=d_model, nhead=nhead,
                              num_layers=num_layers, dim_ff=dim_ff, pad_id=tok.pad_id).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(ignore_index=tok.pad_id)

    for e in range(1, epochs+1):
        model.train()
        opt.zero_grad()
        logits, _ = model(X, pad_mask)
        loss = crit(logits.reshape(-1, logits.size(-1)), Y.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ppl = math.exp(loss.item())
        if e % 5 == 0 or e == 1:
            print(f"epoch {e:02d} | loss {loss.item():.4f} | ppl {ppl:.2f}")
    return model, tok, X, pad_mask

@torch.no_grad()
def generate(model, tok, prompt, max_new_tokens=10, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    ids = tok.encode(prompt).unsqueeze(0).to(device)  # (1,T)
    for _ in range(max_new_tokens):
        pad_mask = (ids == tok.pad_id)
        logits, _ = model(ids, pad_mask)
        next_logits = logits[0, -1]
        probs = torch.softmax(next_logits, dim=-1)
        nxt = torch.multinomial(probs, 1).item()
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
        if nxt == tok.eos_id:
            break
    return tok.decode(ids[0].tolist())

def show_attention_heatmap(model, tok, sentence, device=None):
    """
    Prints a tiny attention matrix for a toy single-head demo on final hidden states.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ids = tok.encode(sentence).unsqueeze(0).to(device)
    pad_mask = (ids == tok.pad_id)
    logits, hidden = model(ids, pad_mask)   # (1,T,V), (1,T,D)
    attn = model.single_head_attention_demo(hidden, head_dim=16)  # (T,T)

    # Pretty print
    tokens = ["<BOS>"] + sentence.split() + ["<EOS>"]
    T = len(tokens)
    print("\nAttention (toy 1-head) for:", sentence)
    print("Rows = query position, Cols = key position")
    header = ["     "] + [f"{i:>5}" for i in range(T)]
    print("".join(header))
    for i in range(T):
        row = [f"{i:>5}"] + [f"{attn[i,j].item():5.2f}" for j in range(T)]
        print("".join(row))
    print("Tokens:", tokens)

if __name__ == "__main__":
    model, tok, X, pad_mask = train(epochs=40, lr=3e-3, d_model=64, nhead=4, num_layers=2, dim_ff=128)

    # Quick demos
    prompts = ["the cat", "the coder", "a frog", "robots"]
    print("\nGenerations:")
    for p in prompts:
        print(f"  {p} … {generate(model, tok, p, max_new_tokens=8)}")

    # Show toy attention on one of the fixed sentences
    show_attention_heatmap(model, tok, "the cat sat on the mat")
    # --- Interactive input round ---
    print("\nInteractive mode (type 'exit' to quit).")
    while True:
        try:
            user_in = input("Prompt> ").strip()
        except EOFError:
            print()
            break
        if not user_in or user_in.lower() in {"exit", "quit", ":q"}:
            break

        # Optionally let the user set max tokens inline like: "the cat || 12"
        if "||" in user_in:
            prompt_text, max_tok_str = map(str.strip, user_in.split("||", 1))
            try:
                max_tok = int(max_tok_str)
            except ValueError:
                print("  (Invalid max token count; using 12)")
                max_tok = 12
        else:
            prompt_text, max_tok = user_in, 12

        out = generate(model, tok, prompt_text, max_new_tokens=max_tok)
        print(f"  {prompt_text} … {out}")
