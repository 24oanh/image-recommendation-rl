import torch.nn as nn

class A2C_Network(nn.Module):
    def __init__(self, encoder, action_dim):
        super(A2C_Network, self).__init__()
        self.encoder = encoder
        
        self.actor = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, action_dim)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        
    def forward(self, state_indices):
        user_rep = self.encoder(state_indices)
        
        logits = self.actor(user_rep)
        value = self.critic(user_rep)
        
        return logits, value
