import os
import sys
import json
import random
import pickle
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.transformer import TransformerEncoder
from models.a2c import A2C_Network
from environment.env import BehanceEnv

# CẤU HÌNH
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

N_STEPS, GAMMA, MAX_EP_A2C = 8, 0.95, 5000
LR_ACTOR, LR_CRITIC = 5e-5, 1e-4
ENTROPY_START, ENTROPY_END = 0.5, 0.05
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def warmup_actor(a2c_net, train_traj, candidate_pool, seq_len=10, pad_token=0):
    """Huấn luyện có giám sát ban đầu giúp Actor dự đoán nhanh hơn"""
    print("Khởi động Actor:")
    optimizer = optim.Adam(a2c_net.parameters(), lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss()
    
    warmup_data = []
    for u, traj in list(train_traj.items())[:5000]:
        for i in range(1, len(traj)):
            state = traj[max(0, i-seq_len):i]
            pad_len = seq_len - len(state)
            state_padded = [pad_token]*pad_len + state
            target_item = traj[i]
            if target_item in candidate_pool:
                target_idx = np.where(candidate_pool == target_item)[0][0]
                warmup_data.append((state_padded, target_idx))
                
    for epoch in range(5):
        random.shuffle(warmup_data)
        total_loss = 0
        for i in range(0, len(warmup_data), 128):
            batch = warmup_data[i:i+128]
            states = torch.tensor([b[0] for b in batch], dtype=torch.long).to(DEVICE)
            targets = torch.tensor([b[1] for b in batch], dtype=torch.long).to(DEVICE)
            
            logits, _ = a2c_net(states)
            loss = criterion(logits, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Warm-up Epoch {epoch+1}/5 | Loss: {total_loss/len(warmup_data):.4f}")

def main():
    print(f"Đang dùng thiết bị: {DEVICE}")
    with open(os.path.join(DATA_DIR, 'train_trajectories.pkl'), 'rb') as f:
        train_traj = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'item_popularity.json'), 'r') as f:
        item_pop = pd.Series(json.load(f)).astype(float)
    import pandas as pd
    
    candidate_pool_idx = np.load(os.path.join(DATA_DIR, 'candidate_pool.npy'))
    image_features_512 = np.load(os.path.join(DATA_DIR, 'image_features_512.npy'))
    pad_token = len(image_features_512)
    
    env = BehanceEnv(train_traj, candidate_pool_idx, image_features_512, item_pop)
    encoder = TransformerEncoder(image_features_512).to(DEVICE)
    a2c_net = A2C_Network(encoder, action_dim=len(candidate_pool_idx)).to(DEVICE)
    
    # Chạy Warm-up
    warmup_actor(a2c_net, train_traj, candidate_pool_idx, pad_token=pad_token)
    
    optimizer_a2c = optim.Adam([
        {'params': a2c_net.actor.parameters(), 'lr': LR_ACTOR},
        {'params': a2c_net.critic.parameters(), 'lr': LR_CRITIC},
        {'params': a2c_net.encoder.parameters(), 'lr': LR_ACTOR}
    ])
    scheduler_a2c = optim.lr_scheduler.CosineAnnealingLR(optimizer_a2c, T_max=MAX_EP_A2C, eta_min=1e-6)
    
    best_reward = -float('inf')
    history_rewards = []
    
    print(f"Bắt đầu huấn luyện A2C ({MAX_EP_A2C} Episodes)...")
    for episode in tqdm(range(1, MAX_EP_A2C + 1), desc="Train A2C"):
        entropy_beta = max(ENTROPY_END, ENTROPY_START - (episode / MAX_EP_A2C) * (ENTROPY_START - ENTROPY_END))
        state = env.reset()
        done, ep_reward = False, 0
        
        while not done:
            log_probs, values, rewards, entropies = [], [], [], []
            
            for _ in range(N_STEPS):
                state_tensor = torch.tensor([state], dtype=torch.long).to(DEVICE)
                logits, value = a2c_net(state_tensor)
                
                # Temperature Scaling
                T = max(0.5, 2.0 - episode / MAX_EP_A2C * 1.5)
                probs = F.softmax(logits / T, dim=-1)
                dist = Categorical(probs)
                
                action_idx = dist.sample()
                log_probs.append(dist.log_prob(action_idx))
                values.append(value)
                entropies.append(dist.entropy())
                
                real_item_idx = env.candidate_pool[action_idx.item()]
                next_state, reward, done, _ = env.step(real_item_idx)
                
                rewards.append(reward)
                ep_reward += reward
                state = next_state
                if done: break
                
            # Tính Returns và Advantage
            R = 0 if done else a2c_net(torch.tensor([state], dtype=torch.long).to(DEVICE))[1].item()
            returns = []
            for r in reversed(rewards):
                R = r + GAMMA * R
                returns.insert(0, R)
                
            returns = torch.tensor(returns, dtype=torch.float32).to(DEVICE)
            values = torch.cat(values).squeeze()
            log_probs = torch.cat(log_probs)
            entropies = torch.cat(entropies)
            
            advantages = returns - values.detach()
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
            actor_loss = -(log_probs * advantages).mean() - entropy_beta * entropies.mean() * 3.0
            critic_loss = F.mse_loss(values, returns)
            total_loss = actor_loss + 0.5 * critic_loss
            
            optimizer_a2c.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(a2c_net.parameters(), max_norm=0.5)
            optimizer_a2c.step()
            
        scheduler_a2c.step()
        history_rewards.append(ep_reward)
        
        if ep_reward > best_reward:
            best_reward = ep_reward
            torch.save(a2c_net.state_dict(), os.path.join(DATA_DIR, 'best_a2c.pth'))
            
        if episode % 100 == 0:
            avg_reward = np.mean(history_rewards[-100:])
            print(f"| A2C Ep: {episode:4d} | Reward TB: {avg_reward:.2f} | Beta: {entropy_beta:.3f} | Entropy: {entropies.mean().item():.2f}")

    print(f"ĐÃ HOÀN TẤT A2C! Best Reward: {best_reward:.2f}")

if __name__ == "__main__":
    main()
