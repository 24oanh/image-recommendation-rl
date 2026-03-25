import torch.nn as nn

class DQN_Network(nn.Module):
    def __init__(self, encoder, action_dim):
        super(DQN_Network, self).__init__()
        self.encoder = encoder
        self.fc1 = nn.Linear(512, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, action_dim) 
        
    def forward(self, state_indices):
        user_rep = self.encoder(state_indices)
        x = self.relu(self.fc1(user_rep))
        q_values = self.fc2(x)                
        return q_values
