"""
model.py — Transformer Architecture Skeleton
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""
## tokenizer in transformer's infer neeeds to be looked into
import math
import copy
import os
import gdown
from typing import Optional, Tuple
import spacy

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

# class LightTokenizer:
#     def __init__(self, nlp):
#         self.tokenizer = nlp.tokenizer
    
#     def __call__(self, text):
#         return self.tokenizer(text)

# ══════════════════════════════════════════════════════════════════════
#   STANDALONE ATTENTION FUNCTION  
#    Exposed at module level so the autograder can import and test it
#    independently of MultiHeadAttention.
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
        mask : Optional Boolean mask, shape broadcastable to
               (..., seq_q, seq_k).
               Positions where mask is True are MASKED OUT
               (set to -inf before softmax).

    Returns:
        output : Attended output,   shape (..., seq_q, d_v)
        attn_w : Attention weights, shape (..., seq_q, seq_k)
    """
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    attn_w = F.softmax(scores, dim=-1)    #dim = -1 helps apply along rows
    attn_w = torch.nan_to_num(attn_w, nan=0.0)
    output = torch.matmul(attn_w, V)
    return output, attn_w
    


# ══════════════════════════════════════════════════════════════════════
# ❷  MASK HELPERS 
#    Exposed at module level so they can be tested independently and
#    reused inside Transformer.forward.
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 0,
) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).

    Args:
        src     : Source token-index tensor, shape [batch, src_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True  → position is a PAD token (will be masked out)
        False → real token
    """
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, src_len]


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 0,
) -> torch.Tensor:
    """
    Build a combined padding + causal (look-ahead) mask for the decoder.

    Args:
        tgt     : Target token-index tensor, shape [batch, tgt_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True → position is masked out (PAD or future token)
    """
   
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, tgt_len]
    seq_len = tgt.size(1)                  
    causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=tgt.device), diagonal=1).bool()  # [tgt_len, tgt_len]
    combined_mask = pad_mask | causal_mask.unsqueeze(0)  # [batch, 1, tgt_len, tgt_len]
    return combined_mask    

# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION 
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.

        MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
        head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)

    You are NOT allowed to use torch.nn.MultiheadAttention.

    Args:
        d_model   (int)  : Total model dimensionality. Must be divisible by num_heads.
        num_heads (int)  : Number of parallel attention heads h.
        dropout   (float): Dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, use_scaling: bool = True) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads   # depth per head
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_scaling = use_scaling
    
    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : shape [batch, seq_q, d_model]
            key   : shape [batch, seq_k, d_model]
            value : shape [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to
                    [batch, num_heads, seq_q, seq_k]
                    True → masked out (attend nowhere)

        Returns:
            output : shape [batch, seq_q, d_model]

        """
        batch_size = query.size(0)
        Q = self.W_q(query)   # (batch, seq_q, d_model)
        K = self.W_k(key)     # (batch, seq_k, d_model)
        V = self.W_v(value)   # (batch, seq_k, d_model)

        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1))

        if self.use_scaling:
            scores = scores / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        self.attn_weights = attn_weights  
        attn_output = torch.matmul(attn_weights, V)
      
        attn_output = self.dropout(attn_output)
        attn_output = attn_output.transpose(1, 2).contiguous()    #tranpose is required so that we dont mix different tokend from differen theads
        attn_output = attn_output.view(batch_size, -1, self.d_model)  
        output = self.W_o(attn_output)
        return output
        #Queries, Keys, Values all have shape [batch, seq_len, d_model]
        #We need to project them to [batch, num_heads, seq_len, d_k]
        #Then we can apply scaled_dot_product_attention to get [batch, num_heads, seq_q, d_k]

        


# ══════════════════════════════════════════════════════════════════════
#   POSITIONAL ENCODING  
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.

    Args:
        d_model  (int)  : Embedding dimensionality.
        dropout  (float): Dropout applied after adding encodings.
        max_len  (int)  : Maximum sequence length to pre-compute (default 5000).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )  # (d_model/2)

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)  # Register as buffer to avoid being treated as a parameter  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input embeddings, shape [batch, seq_len, d_model]

        Returns:
            Tensor of same shape [batch, seq_len, d_model]
            = x  +  PE[:, :seq_len, :]  

        """
         # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

        


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK 
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network, §3.3:

        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

    Args:
        d_model (int)  : Input / output dimensionality (e.g. 512).
        d_ff    (int)  : Inner-layer dimensionality (e.g. 2048).
        dropout (float): Dropout applied between the two linears.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # TODO: Task 2.3 — define:
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : shape [batch, seq_len, d_model]
        Returns:
              shape [batch, seq_len, d_model]
        
        """
        x = self.linear1(x)      # → (batch, seq_len, d_ff)
        x = F.relu(x)            # activation
        x = self.dropout(x)
        x = self.linear2(x)      # → (batch, seq_len, d_model)
        return x
     


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER  
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer:
        x → [Self-Attention → Add & Norm] → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, use_scaling = True) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout,use_scaling)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            shape [batch, src_len, d_model]

        """
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out)) 
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x    


# ══════════════════════════════════════════════════════════════════════
#   DECODER LAYER 
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer:
        x → [Masked Self-Attn → Add & Norm]
          → [Cross-Attn(memory) → Add & Norm]
          → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, use_scaling = True) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout,use_scaling)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout, use_scaling)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : Encoder output, shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            shape [batch, tgt_len, d_model]
        """
        attn1 = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn1))
        attn2 = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout(attn2)) 
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
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : shape [batch, src_len, d_model]
            mask : shape [batch, 1, 1, src_len]
        Returns:
            shape [batch, src_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            shape [batch, tgt_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    


# ══════════════════════════════════════════════════════════════════════
#   FULL TRANSFORMER  
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Args:
        src_vocab_size (int)  : Source vocabulary size.
        tgt_vocab_size (int)  : Target vocabulary size.
        d_model        (int)  : Model dimensionality (default 512).
        N              (int)  : Number of encoder/decoder layers (default 6).
        num_heads      (int)  : Number of attention heads (default 8).
        d_ff           (int)  : FFN inner dimensionality (default 2048).
        dropout        (float): Dropout probability (default 0.1).
    """

    def __init__(
        self,
        src_vocab_size: int = None,
        tgt_vocab_size: int = None,
        d_model:   int   = 512,
        N:         int   = 5,
        num_heads: int   = 8,
        d_ff:      int   = 2048,
        dropout:   float = 0.1,
        use_scaling = True,
        use_positional_encoding=True,
      use_checkpoint = True
    ) -> None:
        super().__init__()
        # TODO: Instantiate 
        # init should also load the model weights if checkpoint path provided, download the .pth file like this
        if src_vocab_size is None or tgt_vocab_size is None:
          base_dir = os.path.dirname(os.path.abspath(__file__))
          vocab = torch.load(os.path.join(base_dir, "vocab.pt"), weights_only=False)
          src_vocab_size = src_vocab_size or len(vocab["src_vocab"])
          tgt_vocab_size = tgt_vocab_size or len(vocab["tgt_vocab"])
          
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model) 
        self.use_pe = use_positional_encoding
        if self.use_pe:
            self.pos_enc = PositionalEncoding(d_model, dropout)
        else:
            self.pos_embed = nn.Embedding(5000, d_model)
        
        encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout,use_scaling)
        decoder_layer = DecoderLayer(d_model, num_heads, d_ff, dropout, use_scaling)

        self.encoder = Encoder(encoder_layer, N)
        self.decoder = Decoder(decoder_layer, N)

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.N = N
        self.num_heads = num_heads
        self.d_ff = d_ff
      
        self.src_vocab = {
    "<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3
}
        self.tgt_vocab = {
            "<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3
        }
        self.src_tokenizer = None
        model_path = os.path.join(base_dir, "best_model.pt")
        if use_checkpoint:
            file_id = "1MFjaY6F_OV0t6mosH9PNEXa03h0IYibE"   
            
            gdown.download(
                id=file_id,
                output=model_path,
                quiet=False
            )
            
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self.load_state_dict(ckpt["model_state_dict"])
        print("Loaded model successfully")
          

    # ── AUTOGRADER HOOKS ── keep these signatures exactly ─────────────

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full encoder stack.

        Args:
            src      : Token indices, shape [batch, src_len]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            memory : Encoder output, shape [batch, src_len, d_model]
        """
        x = self.src_embed(src) * math.sqrt(self.d_model)
        if self.use_pe:
            x = self.pos_enc(x)
        else:
            positions = torch.arange(0, x.size(1)).unsqueeze(0).to(x.device)
            x = x + self.pos_embed(positions)

        # Pass through encoder stack
        memory = self.encoder(x, src_mask)

        return memory

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full decoder stack and project to vocabulary logits.

        Args:
            memory   : Encoder output,  shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt      : Token indices,   shape [batch, tgt_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
      
        if self.use_pe:
            x = self.pos_enc(x)
        else:
            positions = torch.arange(0, x.size(1)).unsqueeze(0).to(x.device)
            x = x + self.pos_embed(positions)
          
        x = self.decoder(x, memory, src_mask, tgt_mask)
        logits = self.fc_out(x)
        return logits

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass.

        Args:
            src      : shape [batch, src_len]
            tgt      : shape [batch, tgt_len]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        memory = self.encode(src, src_mask)
        logits = self.decode(memory, src_mask, tgt, tgt_mask)
        return logits


    def infer(self, src_sentence: str, max_len: int = 50) -> str:
        """
        Translates a German sentence to English using greedy autoregressive decoding.
        
        Args:
            src_sentence: The raw German text.
            max_len: The maximum length of the translated sentence.
            
        Returns:
            The fully translated English string, detokenized and clean.
            """
        self.eval()
        device = next(self.parameters()).device
    
        base_dir = os.path.dirname(os.path.abspath(__file__))
    
        if len(self.src_vocab) <= 4:
            vocab = torch.load(os.path.join(base_dir, "vocab.pt"), weights_only=False)
            self.src_vocab = vocab["src_vocab"]
            self.tgt_vocab = vocab["tgt_vocab"]
    
 
        # tokens = src_sentence.lower().split()
        # # convert to indices
        # src_tokens = []
        # for tok in tokens:
        #     if tok in self.src_vocab:
        #         src_tokens.append(self.src_vocab[tok])
        #     else:
        #         src_tokens.append(self.src_vocab["<unk>"])
        
        # add special tokens
        #src_tokens = [self.src_vocab["<sos>"]] + src_tokens + [self.src_vocab["<eos>"]]

      
        if self.src_tokenizer is None:
          # self.src_tokenizer = torch.load(
          #     os.path.join(base_dir, "tokenizer.pt"), weights_only=False
          # )
          import pickle
          with open(os.path.join(base_dir, "tokenizer.pkl"), "rb") as f:
            self.src_tokenizer = pickle.load(f)
        tokens = [tok.text.lower() for tok in self.src_tokenizer(src_sentence)]
        src_tokens = (
        [self.src_vocab["<sos>"]]
        + [self.src_vocab.get(tok, self.src_vocab["<unk>"]) for tok in tokens]
        + [self.src_vocab["<eos>"]]
         )
      

    
        with torch.no_grad():   # ← critical for speed
            src = torch.tensor(src_tokens).unsqueeze(0).to(device)
            src_mask = make_src_mask(src).to(device)
            memory = self.encode(src, src_mask)
    
            tgt_tokens = [self.tgt_vocab["<sos>"]]
            for _ in range(max_len):
                tgt = torch.tensor(tgt_tokens).unsqueeze(0).to(device)
                tgt_mask = make_tgt_mask(tgt).to(device)
                logits = self.decode(memory, src_mask, tgt, tgt_mask)
                next_token = logits[:, -1, :].argmax(dim=-1).item()
                tgt_tokens.append(next_token)
                if next_token == self.tgt_vocab["<eos>"]:
                    break
    
        itos = {v: k for k, v in self.tgt_vocab.items()}
        output_tokens = []
        for tok in tgt_tokens[1:]:
            if tok == self.tgt_vocab["<eos>"]:
                break
            output_tokens.append(itos.get(tok, "<unk>"))
    
        return " ".join(output_tokens)
    
    
