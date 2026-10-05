"""Transformer from scratch: scaled dot-product attention, MHA, masks, positional encoding,
LayerNorm, residual connections, encoder / decoder-only (GPT) / encoder-decoder, plus a tiny training loop
with AdamW, gradient clipping and a warmup+cosine LR schedule."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask=None):
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    w = F.softmax(scores, dim=-1)
    return w @ v, w


def causal_mask(T, device=None):
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        self.q, self.k, self.v, self.o = (nn.Linear(d_model, d_model) for _ in range(4))

    def _split(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.dk).transpose(1, 2)

    def forward(self, x_q, x_kv=None, mask=None):
        x_kv = x_q if x_kv is None else x_kv
        out, w = scaled_dot_product_attention(self._split(self.q(x_q)), self._split(self.k(x_kv)),
                                              self._split(self.v(x_kv)), mask)
        B, _, T, _ = out.shape
        return self.o(out.transpose(1, 2).reshape(B, T, -1)), w


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class FFN(nn.Module):
    def __init__(self, d, mult=4, p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d * mult), nn.GELU(), nn.Linear(d * mult, d), nn.Dropout(p))

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.attn, self.ffn, self.n1, self.n2 = MultiHeadAttention(d, h), FFN(d), nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x, mask=None):
        x = x + self.attn(self.n1(x), mask=mask)[0]     # pre-norm + residual
        return x + self.ffn(self.n2(x))


class DecoderBlock(nn.Module):
    def __init__(self, d, h, cross=False):
        super().__init__()
        self.self_attn, self.ffn = MultiHeadAttention(d, h), FFN(d)
        self.cross = MultiHeadAttention(d, h) if cross else None
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x, enc=None):
        x = x + self.self_attn(self.n1(x), mask=causal_mask(x.size(1), x.device))[0]
        if self.cross is not None and enc is not None:
            x = x + self.cross(self.n2(x), enc)[0]
        return x + self.ffn(self.n3(x))


class GPT(nn.Module):
    """Decoder-only language model."""

    def __init__(self, vocab, d=128, h=4, layers=4, max_len=256):
        super().__init__()
        self.emb, self.pos = nn.Embedding(vocab, d), PositionalEncoding(d, max_len)
        self.blocks = nn.ModuleList(DecoderBlock(d, h) for _ in range(layers))
        self.norm, self.head = nn.LayerNorm(d), nn.Linear(d, vocab, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx):
        x = self.pos(self.emb(idx))
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))


class Encoder(nn.Module):
    """Encoder-only (BERT-style)."""

    def __init__(self, vocab, d=128, h=4, layers=4):
        super().__init__()
        self.emb, self.pos = nn.Embedding(vocab, d), PositionalEncoding(d)
        self.blocks = nn.ModuleList(EncoderBlock(d, h) for _ in range(layers))

    def forward(self, idx, mask=None):
        x = self.pos(self.emb(idx))
        for b in self.blocks:
            x = b(x, mask)
        return x


class Seq2Seq(nn.Module):
    """Encoder-decoder (T5 / original Transformer style)."""

    def __init__(self, vocab, d=128, h=4, layers=2):
        super().__init__()
        self.enc = Encoder(vocab, d, h, layers)
        self.emb, self.pos = nn.Embedding(vocab, d), PositionalEncoding(d)
        self.dec = nn.ModuleList(DecoderBlock(d, h, cross=True) for _ in range(layers))
        self.head = nn.Linear(d, vocab)

    def forward(self, src, tgt):
        enc = self.enc(src)
        x = self.pos(self.emb(tgt))
        for b in self.dec:
            x = b(x, enc)
        return self.head(x)


def demo_train(steps=300):
    """Next-token prediction on a tiny repeating corpus."""
    text = "attention is all you need " * 200
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text])
    model = GPT(len(chars), d=64, h=4, layers=2, max_len=64)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    warm = 20
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, 0.5 * (1 + math.cos(math.pi * s / steps))))
    for s in range(steps):
        ix = torch.randint(0, len(data) - 33, (32,))
        x = torch.stack([data[i:i + 32] for i in ix])
        y = torch.stack([data[i + 1:i + 33] for i in ix])
        loss = F.cross_entropy(model(x).reshape(-1, len(chars)), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if s % 50 == 0:
            print(f"step {s:3d} loss {loss.item():.4f}")


if __name__ == "__main__":
    demo_train()
