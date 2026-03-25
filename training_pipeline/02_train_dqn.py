import os
import sys
import json
import random
import pickle
import copy
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.transformer import TransformerEncoder
from models.dqn import DQN_Network
from environment.env import BehanceEnv

# CẤU HÌNH
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

BATCH_SIZE, GAMMA, LR, BUFFER_SIZE = 256, 0.95, 3e-4, 100_000
MAX_EPISODES, TARGET_UPDATE, WARMUP_STEPS = 5000, 100, 2000
EPSILON_START, EPSILON_END, EPSILON_DECAY = 1.0, 0.05, 0.997
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=50000):
        self.capacity, self.alpha = capacity, alpha
        self.beta_start, self.beta_frames = beta_start, beta_frames
        self.frame, self.pos = 1, 0
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)

    def push(self, state, action_idx, reward, next_state, done):
        max_p = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action_idx, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action_idx, reward, next_state, done)
        self.priorities[self.pos] = max_p
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        n = len(self.buffer)
        probs = self.priorities[:n] ** self.alpha
        probs /= probs.sum()
        idxs = np.random.choice(n, batch_size, replace=False, p=probs)
        
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1
        w = (n * probs[idxs]) ** (-beta)
        w /= w.max()
        
        batch = [self.buffer[i] for i in idxs]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), 
                np.array(next_states), np.array(dones, dtype=np.float32), idxs, np.array(w, dtype=np.float32))

    def update_priorities(self, idxs, td_errors):
        for i, e in zip(idxs, td_errors):
            self.priorities[i] = abs(e) + 1e-6

def main():
    print(f"Đang dùng thiết bị: {DEVICE}")
    # 1. Load Data
    with open(os.path.join(DATA_DIR, 'train_trajectories.pkl'), 'rb') as f:
        train_traj = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'item_popularity.json'), 'r') as f:
        item_pop = pd.Series(json.load(f)).astype(float) # Dùng pandas cho môi trường
    import pandas as pd
    
    candidate_pool_idx = np.load(os.path.join(DATA_DIR, 'candidate_pool.npy'))
    image_features_512 = np.load(os.path.join(DATA_DIR, 'image_features_512.npy'))
    
    # 2. Init Env & Models
    env = BehanceEnv(train_traj, candidate_pool_idx, image_features_512, item_pop)
    encoder = TransformerEncoder(image_features_512).to(DEVICE)
    dqn_net = DQN_Network(encoder, action_dim=len(candidate_pool_idx)).to(DEVICE)
    target_net = copy.deepcopy(dqn_net).to(DEVICE)
    
    optimizer = optim.AdamW(dqn_net.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPISODES, eta_min=1e-5)
    buffer = PrioritizedReplayBuffer(BUFFER_SIZE)
    epsilon = EPSILON_START
    
    best_reward = -float('inf')
    history_rewards = []
    
    # 3. Training Loop
    print(f"Bắt đầu huấn luyện DQN ({MAX_EPISODES} Episodes)...")
    for episode in tqdm(range(1, MAX_EPISODES + 1), desc="Train DQN"):
        state = env.reset()
        ep_reward, ep_loss, steps = 0, 0, 0
        
        while True:
            if random.random() < epsilon:
                action_idx = random.randrange(len(candidate_pool_idx))
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor([state], dtype=torch.long).to(DEVICE)
                    action_idx = dqn_net(state_tensor).argmax().item()
            
            real_item_idx = env.candidate_pool[action_idx]
            next_state, reward, done, _ = env.step(real_item_idx)
            buffer.push(state, action_idx, reward, next_state, done)
            
            state = next_state
            ep_reward += reward
            steps += 1
            
            if len(buffer.buffer) >= max(BATCH_SIZE, WARMUP_STEPS):
                b_states, b_actions, b_rewards, b_next_states, b_dones, idxs, w = buffer.sample(BATCH_SIZE)
                
                b_states = torch.tensor(b_states, dtype=torch.long).to(DEVICE)
                b_actions = torch.tensor(b_actions, dtype=torch.long).unsqueeze(1).to(DEVICE)
                b_rewards = torch.tensor(b_rewards).unsqueeze(1).to(DEVICE)
                b_next_states = torch.tensor(b_next_states, dtype=torch.long).to(DEVICE)
                b_dones = torch.tensor(b_dones).unsqueeze(1).to(DEVICE)
                w = torch.tensor(w).unsqueeze(1).to(DEVICE)
                
                current_q = dqn_net(b_states).gather(1, b_actions)
                with torch.no_grad():
                    best_actions = dqn_net(b_next_states).argmax(1, keepdim=True)
                    max_next_q = target_net(b_next_states).gather(1, best_actions)
                    target_q = b_rewards + (GAMMA * max_next_q * (1 - b_dones))
                
                td_errors = (current_q - target_q).detach().cpu().numpy().flatten()
                buffer.update_priorities(idxs, td_errors)
                
                loss = (w * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(dqn_net.parameters(), max_norm=1.0)
                optimizer.step()
                ep_loss += loss.item()
                
            if done: break
            
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        scheduler.step()
        history_rewards.append(ep_reward)
        
        # Save Best Model & Update Target
        if ep_reward > best_reward:
            best_reward = ep_reward
            torch.save(dqn_net.state_dict(), os.path.join(DATA_DIR, 'best_dqn.pth'))
            
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(dqn_net.state_dict())
            
        if episode % 100 == 0:
            avg_reward_100 = np.mean(history_rewards[-100:])
            print(f"| DQN Ep: {episode:4d} | Reward TB: {avg_reward_100:.2f} | Eps: {epsilon:.3f}")

    print(f"ĐÃ HOÀN TẤT! Best Reward: {best_reward:.2f}")

if __name__ == "__main__":
    main()
