import torch
import torch.nn as nn
import numpy as np

class TransformerEncoder(nn.Module):
    def __init__(self, feature_matrix, hidden_size=512, n_heads=4, n_layers=2, seq_len=10):
        super().__init__()
        # Tạo vector đệm (pad_vector) cho index 0
        pad_vector = np.zeros((1, feature_matrix.shape[1]), dtype=np.float32)
        full_features = np.vstack([feature_matrix, pad_vector])
        
        # Embedding cho đặc trưng ảnh và ID
        self.img_embed = nn.Embedding.from_pretrained(torch.tensor(full_features), freeze=True)
        self.id_embed = nn.Embedding(len(full_features), 256)
        
        self.project = nn.Linear(512 + 256, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, hidden_size))
        
        # Lớp Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=n_heads, 
            dim_feedforward=1024, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(self, state_indices):
        img_x = self.img_embed(state_indices) 
        id_x = self.id_embed(state_indices)   
        x = torch.cat([img_x, id_x], dim=-1)  
        x = self.project(x)                   
        x = x + self.pos_embed                
        x = self.transformer(x)               
        x = x[:, -1, :]
        x = self.out_proj(x)
        return x
