# model_def.py
import torch
import torch.nn as nn
import numpy as np

# --- Multi-Head Self-Attention Module ---
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(context)
        return output

# --- Transformer Layer ---
class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# --- Phishing Transformer Model Definition ---
class PhishingTransformerModel(nn.Module):
    # Added max_seq_len parameter
    def __init__(self, input_dim, max_seq_len, embed_dim=64, num_heads=4, num_layers=2, ff_dim=128, dropout=0.1, num_classes=1):
        super(PhishingTransformerModel, self).__init__()
        self.embedding = nn.Linear(1, embed_dim) # Project each feature 'token'

        # Use the passed max_seq_len for positional encoding buffer
        self.register_buffer("pos_encoding", self._get_positional_encoding(max_seq_len, embed_dim))

        self.transformer_layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, num_classes),
            nn.Sigmoid()
        )
        self.dropout_layer = nn.Dropout(dropout)

    def _get_positional_encoding(self, max_seq_len, d_model):
        # Standard positional encoding formula
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0) # Add batch dimension

    def forward(self, x):
        # x shape: [batch_size, input_dim (number of features)]
        batch_size, seq_len = x.size() # seq_len here is the number of features

        # Treat each feature as a 'token' in a sequence: Reshape to [batch_size, seq_len, 1]
        x = x.unsqueeze(2)

        # Embed each feature 'token'
        x = self.embedding(x) # Output shape: [batch_size, seq_len, embed_dim]

        # Add positional encoding - slice it to match the input sequence length (number of features)
        # Ensure positional encoding length is sufficient
        if seq_len > self.pos_encoding.size(1):
             raise ValueError(f"Input sequence length ({seq_len}) exceeds positional encoding max length ({self.pos_encoding.size(1)})")
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout_layer(x)

        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Global average pooling over the sequence dimension (features dimension)
        x = torch.mean(x, dim=1) # Output shape: [batch_size, embed_dim]

        # Classification
        x = self.classifier(x) # Output shape: [batch_size, num_classes]

        return x