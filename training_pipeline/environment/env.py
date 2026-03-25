import numpy as np
import random

class BehanceEnv:
    """
    Môi trường mô phỏng tương tác người dùng Behance.
    Mỗi Step, Agent gợi ý 1 ảnh và nhận lại Reward dựa trên:
    - Độ chính xác (Semantic Match/Exact Match)
    - Tính mới mẻ (Novelty)
    - Đa dạng (Diversity)
    - Tránh lặp lại (Repetition Penalty)
    """
    def __init__(self, trajectories, candidate_pool_idx, image_features, item_popularity_train, seq_len=10):
        self.trajectories = trajectories
        self.candidate_pool = np.array(candidate_pool_idx, dtype=np.int32)
        self.seq_len = seq_len
        
        # Tiền xử lý: Tính L2 Norm cho đặc trưng ảnh để tính Cosine Similarity nhanh hơn
        norms = np.linalg.norm(image_features, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.norm_features = image_features / norms
        
        # Tiền xử lý: Độ phổ biến (Popularity) cho các item trong candidate pool
        raw_pops = item_popularity_train.reindex(candidate_pool_idx).fillna(0).values
        self.pop_scores = raw_pops / (raw_pops.max() + 1e-10)
        self.item_to_pop = dict(zip(candidate_pool_idx, self.pop_scores))
        
        # Các thông số hệ thống
        self.n_items_total = len(image_features)
        self.pad_token = self.n_items_total
        # Chỉ giữ lại người dùng có ít nhất 2 tương tác trong lịch sử
        self.valid_users = [u for u, traj in trajectories.items() if len(traj) >= 2]
        
        # Theo dõi độ bao phủ (Coverage)
        self.recommend_freq = np.zeros(len(candidate_pool_idx), dtype=np.float32)
        self.total_steps = 0
        
    def reset(self, user_idx=None):
        """Khởi tạo lại môi trường với một user mới."""
        if user_idx is None:
            user_idx = random.choice(self.valid_users)
        self.current_user = user_idx
        self.current_traj = self.trajectories[user_idx]
        self.t = 1
        return self._get_state()
        
    def _get_state(self):
        """Trích xuất lịch sử tương tác và đệm (padding) thành State."""
        history = self.current_traj[max(0, self.t - self.seq_len) : self.t]
        pad_len = self.seq_len - len(history)
        return np.array([self.pad_token] * pad_len + history, dtype=np.int32)
        
    def step(self, action_item_idx):
        """Thực hiện một hành động (gợi ý ảnh) và trả về Reward."""
        target_item = self.current_traj[self.t]
        reward = 0.0
        
        # 1. Exact match / Semantic match (Cosine)
        if action_item_idx == target_item:
            reward += 15.0
        else:
            cos_sim = np.dot(self.norm_features[action_item_idx], self.norm_features[target_item])
            reward += max(0.0, cos_sim) * 6.0
            
        # 2. Novelty (Điểm cao hơn cho những item ít phổ biến)
        pop = self.item_to_pop.get(action_item_idx, 0.0)
        reward += (1.0 - pop) * 0.5
        
        # 3. Diversity (Khuyến khích khác biệt so với item vừa xem ngay trước đó)
        if self.t >= 2:
            prev_item = self.current_traj[self.t - 1]
            div_sim = np.dot(self.norm_features[action_item_idx], self.norm_features[prev_item])
            reward += max(0.0, 1.0 - div_sim) * 0.3
            
        # 4. Penalty: Tránh đề xuất lại những item vừa xem trong 5 bước gần nhất
        recent = self.current_traj[max(0, self.t - 5): self.t]
        if action_item_idx in recent:
            pos = list(reversed(recent)).index(action_item_idx)
            reward -= 3.0 * (1.0 + pos / 5.0)
            
        # 5. Khuyến khích Coverage (Gợi ý các item chưa được lên sóng nhiều)
        pool_positions = np.where(self.candidate_pool == action_item_idx)[0]
        if len(pool_positions) > 0:
            pidx = pool_positions[0]
            freq = self.recommend_freq[pidx] / (self.total_steps + 1)
            reward += (1.0 - freq) * 1.5
            self.recommend_freq[pidx] += 1
            
        self.total_steps += 1
        self.t += 1
        
        # Kiểm tra điều kiện kết thúc chuỗi
        done = (self.t >= len(self.current_traj))
        next_state = self._get_state() if not done else np.array([self.pad_token] * self.seq_len)
        reward = np.clip(reward, -10, 15)
        
        return next_state, reward, done, {"target": target_item}
