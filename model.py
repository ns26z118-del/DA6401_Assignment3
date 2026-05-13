"""
model.py — Transformer Architecture
DA6401 Assignment 3: "Attention Is All You Need"
"""

import math
import copy
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#  SCALED DOT-PRODUCT ATTENTION
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.
        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : Query tensor,  shape (..., seq_q, d_k)
        K    : Key tensor,    shape (..., seq_k, d_k)
        V    : Value tensor,  shape (..., seq_k, d_v)
        mask : Optional Boolean mask, broadcastable to (..., seq_q, seq_k).
               True positions are MASKED OUT (set to -inf before softmax).

    Returns:
        output : shape (..., seq_q, d_v)
        attn_w : shape (..., seq_q, seq_k)
    """
    d_k = Q.size(-1)
    # Scaled scores: (..., seq_q, seq_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    attn_w = F.softmax(scores, dim=-1)
    # Replace NaN from all-masked rows (softmax of all -inf) with 0
    attn_w = torch.nan_to_num(attn_w, nan=0.0)

    output = torch.matmul(attn_w, V)
    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
#  MASK HELPERS
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """
    Build a padding mask for the encoder.

    Args:
        src     : shape [batch, src_len]
        pad_idx : index of <pad>

    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True → PAD (masked out), False → real token
    """
    # (batch, 1, 1, src_len)
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """
    Build a combined padding + causal mask for the decoder.

    Args:
        tgt     : shape [batch, tgt_len]
        pad_idx : index of <pad>

    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True → masked out
    """
    batch_size, tgt_len = tgt.size()

    # Padding mask: (batch, 1, 1, tgt_len) → broadcast to (batch,1,tgt_len,tgt_len)
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)  # (batch,1,1,tgt_len)

    # Causal mask: upper triangle excluding diagonal → (1, 1, tgt_len, tgt_len)
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=tgt.device), diagonal=1
    ).unsqueeze(0).unsqueeze(0)

    return pad_mask | causal_mask


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in §3.2.2.
    NOT using torch.nn.MultiheadAttention.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.attn_weights = None  # store for visualization

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : [batch, seq_q, d_model]
            key   : [batch, seq_k, d_model]
            value : [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to [batch, num_heads, seq_q, seq_k]

        Returns:
            output : [batch, seq_q, d_model]
        """
        batch_size = query.size(0)

        # Linear projections and reshape: (batch, seq, d_model) → (batch, h, seq, d_k)
        def project_and_split(linear, x):
            return linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        Q = project_and_split(self.W_q, query)  # (batch, h, seq_q, d_k)
        K = project_and_split(self.W_k, key)    # (batch, h, seq_k, d_k)
        V = project_and_split(self.W_v, value)  # (batch, h, seq_k, d_k)

        # Scaled dot-product attention
        x, self.attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        # x: (batch, h, seq_q, d_k)

        # Apply dropout to attention weights (re-weight V instead)
        # Standard: dropout on attn weights before multiplying V
        # Here we already have output; common practice is dropout inside sdpa
        # We'll apply dropout on x for regularization:
        x = self.dropout(x)

        # Concat heads: (batch, h, seq_q, d_k) → (batch, seq_q, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.W_o(x)


# ══════════════════════════════════════════════════════════════════════
#  POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding, §3.5.
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build PE table: (1, max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        # Register as buffer — not a trainable parameter
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [batch, seq_len, d_model]
        Returns:
            [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """FFN(x) = max(0, xW₁+b₁)W₂+b₂"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout  = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    x → [Self-Attention → Add & Norm] → [FFN → Add & Norm]
    Uses Post-LayerNorm (original paper).
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # Self-attention sub-layer with residual
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN sub-layer with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


# ══════════════════════════════════════════════════════════════════════
#  DECODER LAYER
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    x → [Masked Self-Attn → Add & Norm]
      → [Cross-Attn(memory) → Add & Norm]
      → [FFN → Add & Norm]
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(p=dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Masked self-attention
        attn1 = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn1))
        # Cross-attention over encoder memory
        attn2 = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout(attn2))
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.self_attn.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.self_attn.d_model)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#  FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model:   int   = 512,
        N:         int   = 6,
        num_heads: int   = 8,
        d_ff:      int   = 2048,
        dropout:   float = 0.1,
        checkpoint_path: str = None,
    ) -> None:
        super().__init__()

        self.d_model = d_model

        # Embeddings
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, dropout)

        # Encoder & Decoder stacks
        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)

        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        # Weight tying (optional but beneficial): tie tgt embedding & output projection
        # self.output_projection.weight = self.tgt_embed.weight

        self._init_weights()

        # Store config for checkpointing
        self.model_config = {
            'src_vocab_size': src_vocab_size,
            'tgt_vocab_size': tgt_vocab_size,
            'd_model':   d_model,
            'N':         N,
            'num_heads': num_heads,
            'd_ff':      d_ff,
            'dropout':   dropout,
        }

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location='cpu')
            self.load_state_dict(state['model_state_dict'])

    def _init_weights(self):
        """Xavier uniform initialization for linear layers."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── AUTOGRADER HOOKS ────────────────────────────────────────────

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src      : [batch, src_len]
            src_mask : [batch, 1, 1, src_len]
        Returns:
            memory : [batch, src_len, d_model]
        """
        x = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            memory   : [batch, src_len, d_model]
            src_mask : [batch, 1, 1, src_len]
            tgt      : [batch, tgt_len]
            tgt_mask : [batch, 1, tgt_len, tgt_len]
        Returns:
            logits : [batch, tgt_len, tgt_vocab_size]
        """
        x = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.output_projection(x)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            src      : [batch, src_len]
            tgt      : [batch, tgt_len]
            src_mask : [batch, 1, 1, src_len]
            tgt_mask : [batch, 1, tgt_len, tgt_len]
        Returns:
            logits : [batch, tgt_len, tgt_vocab_size]
        """
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def infer(self, src_sentence: str) -> str:
        """
        Translate a German sentence to English using greedy decoding.
        Requires self.src_vocab, self.tgt_vocab, self.spacy_de to be set
        externally after training (via dataset setup).
        """
        import spacy
        self.eval()
        nlp_de = spacy.load('de_core_news_sm')
        tokens = [tok.text.lower() for tok in nlp_de(src_sentence)]

        # Use stored vocab objects set externally
        sos_idx = self.src_vocab['<sos>']
        eos_idx = self.src_vocab['<eos>']
        pad_idx = self.src_vocab['<pad>']
        unk_idx = self.src_vocab['<unk>']

        indices = [sos_idx] + [self.src_vocab.get(t, unk_idx) for t in tokens] + [eos_idx]
        src = torch.tensor(indices).unsqueeze(0)
        src_mask = make_src_mask(src, pad_idx)

        from train import greedy_decode
        tgt_sos = self.tgt_vocab['<sos>']
        tgt_eos = self.tgt_vocab['<eos>']
        with torch.no_grad():
            ys = greedy_decode(self, src, src_mask, max_len=100,
                               start_symbol=tgt_sos, end_symbol=tgt_eos)

        tgt_itos = {v: k for k, v in self.tgt_vocab.items()}
        tokens_out = [tgt_itos.get(i, '<unk>') for i in ys.squeeze().tolist()]
        # Strip <sos> and <eos>
        tokens_out = [t for t in tokens_out if t not in ('<sos>', '<eos>', '<pad>')]
        return ' '.join(tokens_out)