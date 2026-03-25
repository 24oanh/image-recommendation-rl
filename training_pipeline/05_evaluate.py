import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.transformer import TransformerEncoder
from models.dqn import DQN_Network
from models.a2c import A2C_Network
from environment.env import BehanceEnv

# CẤU HÌNH
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_ndcg(rank):
    """Tính Discounted Cumulative Gain."""
    return 1 / np.log2(rank + 2)

def evaluate_model(model, model_type, test_trajectories, env, k=10):
    """Đánh giá mô hình trên tập Test theo các chỉ số chuẩn của Hệ thống gợi ý."""
    hits, ndcgs = [], []
    recommended_items = set()
    global_item_freq = np.zeros(len(env.candidate_pool), dtype=np.float32)
    total_users_evaluated = 0

    if hasattr(model, 'eval'):
        model.eval()

    print(f"Đánh giá {model_type} trên {len(test_trajectories):,} users...")

    for user_idx, traj in test_trajectories.items():
        if len(traj) < 2: continue

        # Chia history (để dự đoán) và target (đáp án)
        history = traj[:-1]
        target = traj[-1]

        state = history[-env.seq_len:]
        pad_len = env.seq_len - len(state)
        if pad_len > 0:
            state = [env.pad_token] * pad_len + state

        state_tensor = torch.tensor([state], dtype=torch.long).to(DEVICE)

        with torch.no_grad():
            if model_type == "DQN":
                scores = model(state_tensor).squeeze().cpu().numpy()
            elif model_type == "A2C":
                logits, _ = model(state_tensor)
                scores = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
                # A2C kết hợp thêm yếu tố tránh đề xuất item quá phổ biến
                novelty = 1.0 - global_item_freq / (total_users_evaluated + 1)
                scores = scores + 0.3 * novelty
            elif model_type == "Popularity":
                scores = env.pop_scores.copy()
            else: # Random
                scores = np.random.rand(len(env.candidate_pool))

        # Xóa các item đã xem trong lịch sử khỏi danh sách gợi ý
        history_set = set(history)
        for i, item_idx in enumerate(env.candidate_pool):
            if item_idx in history_set:
                scores[i] = -999999 

        # Lấy Top-K
        top_k_indices = np.argsort(scores)[-k:][::-1]
        top_k_items = [env.candidate_pool[idx] for idx in top_k_indices]

        # Cập nhật Coverage & Freq
        for idx in top_k_indices:
            global_item_freq[idx] += 1
        recommended_items.update(top_k_items)
        total_users_evaluated += 1

        # Kiểm tra Hit và NDCG
        if target in top_k_items:
            hits.append(1)
            rank = top_k_items.index(target)
            ndcgs.append(get_ndcg(rank))
        else:
            hits.append(0)
            ndcgs.append(0)

    hr = np.mean(hits)
    ndcg = np.mean(ndcgs)
    cov = (len(recommended_items) / len(env.candidate_pool)) * 100

    return hr, ndcg, cov

def main():
    print(" ĐÁNH GIÁ MÔ HÌNH: ")
    
    # 1. Load Data
    with open(os.path.join(DATA_DIR, 'test_trajectories.pkl'), 'rb') as f:
        test_traj = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'item_popularity.json'), 'r') as f:
        item_pop_dict = json.load(f)
        item_pop = pd.Series(item_pop_dict).astype(float)
        
    candidate_pool_idx = np.load(os.path.join(DATA_DIR, 'candidate_pool.npy'))
    image_features_512 = np.load(os.path.join(DATA_DIR, 'image_features_512.npy'))
    
    # Khởi tạo Env ảo để mượn cấu hình
    env = BehanceEnv(test_traj, candidate_pool_idx, image_features_512, item_pop)
    
    # 2. Load Models
    encoder_dqn = TransformerEncoder(image_features_512).to(DEVICE)
    dqn_model = DQN_Network(encoder_dqn, action_dim=len(candidate_pool_idx)).to(DEVICE)
    
    encoder_a2c = TransformerEncoder(image_features_512).to(DEVICE)
    a2c_model = A2C_Network(encoder_a2c, action_dim=len(candidate_pool_idx)).to(DEVICE)
    
    try:
        dqn_model.load_state_dict(torch.load(os.path.join(DATA_DIR, 'best_dqn.pth'), map_location=DEVICE))
        a2c_model.load_state_dict(torch.load(os.path.join(DATA_DIR, 'best_a2c.pth'), map_location=DEVICE), strict=False)
        print("Đã load trọng số huấn luyện thành công!\n")
    except Exception as e:
        print(f"Lỗi load weights: {e}")
        return

    # 3. Đánh giá
    results = []
    models_to_evaluate = [
        (None, "Random"),
        (None, "Popularity"),
        (dqn_model, "DQN"),
        (a2c_model, "A2C")
    ]
    
    for model, name in models_to_evaluate:
        hr, ndcg, cov = evaluate_model(model, name, test_traj, env)
        results.append({
            "Model": name,
            "HR@10": f"{hr:.6f}",
            "NDCG@10": f"{ndcg:.6f}",
            "Coverage (%)": f"{cov:.2f}%"
        })

    # 4. Hiển thị bảng kết quả
    print(" BẢNG KẾT QUẢ TỔNG HỢP TRÊN TẬP TEST: ")
    df_results = pd.DataFrame(results)
    print(df_results.to_markdown(index=False))

if __name__ == "__main__":
    main()
