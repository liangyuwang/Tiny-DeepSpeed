# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0

import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from dataclasses import dataclass

# modified from https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py

# -----------------------------------------------------------------------------
# Model definition

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    max_vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    attention = "standard_attention" # "standard_attention", "flash_attention"

# Masked Multi-Head Self-Attention

def standard_attention(q, k, v, dropout=True, dropout_p=0.0):
    T = q.size(-3)
    scale = 1.0 / math.sqrt(k.size(-1))
    q = q.transpose(1, 2) # (B, nh, T, hs)
    k = k.transpose(1, 2) # (B, nh, T, hs)
    v = v.transpose(1, 2) # (B, nh, T, hs)
    # manual implementation of attention
    att = (q @ k.transpose(-2, -1)) * scale
    mask = torch.tril(torch.ones(T, T, device=v.device)).view(1, 1, T, T)
    att = att.masked_fill(mask[:,:,:T,:T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    att = F.dropout(att, p=dropout_p, training=dropout)
    y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    return y.transpose(1, 2).contiguous() # (B, T, nh, hs)

def flash_attention(q, k, v, dropout=True, dropout_p=0.0):
    scale = 1.0 / math.sqrt(k.size(-1))
    if not dropout: dropout_p=0.0
    q = q.transpose(1, 2) # (B, nh, T, hs)
    k = k.transpose(1, 2) # (B, nh, T, hs)
    v = v.transpose(1, 2) # (B, nh, T, hs)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale, dropout_p=dropout_p)
    return y.transpose(1, 2).contiguous() # (B, T, nh, hs)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.attention = config.attention
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head) # (B, T, nh, hs)
        q = q.view(B, T, self.n_head, C // self.n_head) # (B, T, nh, hs)
        v = v.view(B, T, self.n_head, C // self.n_head) # (B, T, nh, hs)
        # Attention implementation
        if self.attention == "standard_attention":
            y = standard_attention(q, k, v, dropout=self.training, dropout_p=False)
        elif self.attention == "flash_attention":
            y = flash_attention(q, k, v, dropout=self.training, dropout_p=False)
        y = y.view(B, T, C)
        # output projection
        y = self.c_proj(y)
        return y

# Feed-Forward Neural Network

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

# Transformer Block

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        out1 = self.ln_1(x)
        out1 = self.attn(out1)
        x = x + out1
        out2 = self.ln_2(x)
        out2 = self.mlp(out2)
        x = x + out2
        return x

# GPT-2 Model

class GPT2Model(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
