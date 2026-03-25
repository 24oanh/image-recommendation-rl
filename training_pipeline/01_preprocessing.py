import os
import json
import pickle
import struct
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. CẤU HÌNH ĐƯỜNG DẪN VÀ THAM SỐ
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
SAVE_DIR = os.path.join(BASE_DIR, 'data', 'processed')
os.makedirs(SAVE_DIR, exist_ok=True)

INTERACTION_FILE = os.path.join(RAW_DIR, 'Behance_appreciate_1M.txt')
IMAGE_FEAT_FILE = os.path.join(RAW_DIR, 'Behance_Image_Features.b')

K_CORE = 10
CANDIDATE_POOL_SIZE = 500
PCA_DIM = 512
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15

def apply_kcore(df, k=10, max_iter=20):
    """Lọc những User và Item có ít nhất k tương tác."""
    for i in range(max_iter):
        before_len = len(df)
        user_cnt = df.groupby('user_id').size()
        item_cnt = df.groupby('item_id').size()
        
        valid_users = user_cnt[user_cnt >= k].index
        valid_items = item_cnt[item_cnt >= k].index
        
        df = df[df['user_id'].isin(valid_users) & df['item_id'].isin(valid_items)]
        print(f"  [Vòng {i+1}] Còn lại: {len(df):,} tương tác")
        if len(df) == before_len:
            print(f"  Hội tụ tại vòng {i+1}!")
            break
    return df.reset_index(drop=True)

def load_image_features(filepath):
    """Đọc file binary đặc trưng ảnh nguyên thủy (4096 dims)."""
    features = {}
    with open(filepath, 'rb') as f:
        count = 0
        while True:
            raw_id = f.read(8)
            if len(raw_id) < 8: break
            item_id = raw_id.decode('ascii').strip('\x00')
            raw_feat = f.read(4096 * 4)
            if len(raw_feat) < 4096 * 4: break
            feat = struct.unpack('4096f', raw_feat)
            features[item_id] = np.array(feat, dtype=np.float32)
            count += 1
            if count % 50000 == 0:
                print(f" Đã load {count:,} items")
    return features

def main():
    df = pd.read_csv(INTERACTION_FILE, sep=r'\s+', header=None, names=['user_id', 'item_id', 'timestamp'])
    df_filtered = apply_kcore(df, k=K_CORE)
    
    # Loại bỏ các item không có đặc trưng ảnh
    features_raw = load_image_features(IMAGE_FEAT_FILE)
    df_filtered['item_id_str'] = df_filtered['item_id'].astype(str).str.zfill(8)
    items_with_features = set(features_raw.keys())
    df_filtered = df_filtered[df_filtered['item_id_str'].isin(items_with_features)].copy()

    item_popularity = df_filtered.groupby('item_id_str').size().sort_values(ascending=False)
    candidate_pool_raw = item_popularity.head(CANDIDATE_POOL_SIZE).index.tolist()
    
    # Chuyển đổi ID sang Index liên tục
    unique_users = sorted(df_filtered['user_id'].unique())
    unique_items = sorted(df_filtered['item_id_str'].unique())
    user2idx = {u: i for i, u in enumerate(unique_users)}
    item2idx = {it: i for i, it in enumerate(unique_items)}
    
    df_filtered['user_idx'] = df_filtered['user_id'].map(user2idx)
    df_filtered['item_idx'] = df_filtered['item_id_str'].map(item2idx)
    candidate_pool_idx = [item2idx[it] for it in candidate_pool_raw]
    
    # Lưu Mappings
    with open(os.path.join(SAVE_DIR, 'user_mappings.json'), 'w') as f:
        json.dump({str(k): int(v) for k, v in user2idx.items()}, f)
    with open(os.path.join(SAVE_DIR, 'idx2item.json'), 'w') as f:
        json.dump({str(v): str(k) for k, v in item2idx.items()}, f)
    np.save(os.path.join(SAVE_DIR, 'candidate_pool.npy'), np.array(candidate_pool_idx))
    with open(os.path.join(SAVE_DIR, 'item_popularity.json'), 'w') as f:
        json.dump({str(k): int(v) for k, v in item_popularity.to_dict().items()}, f)

    n_items_total = len(item2idx)
    feature_matrix = np.zeros((n_items_total, 4096), dtype=np.float32)
    for it_str, idx in tqdm(item2idx.items(), desc="Chuẩn bị Ma trận"):
        feature_matrix[idx] = features_raw[it_str]
        
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)
    
    pca = PCA(n_components=PCA_DIM, random_state=42)
    image_features_512 = pca.fit_transform(feature_matrix_scaled)
    np.save(os.path.join(SAVE_DIR, 'image_features_512.npy'), image_features_512)
    print(f"  Đã lưu image_features_512.npy. Giữ lại {pca.explained_variance_ratio_.sum()*100:.2f}% thông tin.")

    df_filtered = df_filtered.sort_values(['user_idx', 'timestamp']).reset_index(drop=True)
    df_filtered['rank'] = df_filtered.groupby('user_idx').cumcount()
    df_filtered['total'] = df_filtered.groupby('user_idx')['user_idx'].transform('count')
    df_filtered['pct'] = df_filtered['rank'] / df_filtered['total']
    
    df_train = df_filtered[df_filtered['pct'] < TRAIN_RATIO]
    df_val = df_filtered[(df_filtered['pct'] >= TRAIN_RATIO) & (df_filtered['pct'] < TRAIN_RATIO + VAL_RATIO)]
    df_test = df_filtered[df_filtered['pct'] >= TRAIN_RATIO + VAL_RATIO]
    
    def extract_traj(df):
        return df.groupby('user_idx')['item_idx'].apply(list).to_dict()
        
    with open(os.path.join(SAVE_DIR, 'train_trajectories.pkl'), 'wb') as f:
        pickle.dump(extract_traj(df_train), f)
    with open(os.path.join(SAVE_DIR, 'val_trajectories.pkl'), 'wb') as f:
        pickle.dump(extract_traj(df_val), f)
    with open(os.path.join(SAVE_DIR, 'test_trajectories.pkl'), 'wb') as f:
        pickle.dump(extract_traj(df_test), f)
        
    print("HOÀN TẤT TOÀN BỘ QUÁ TRÌNH TIỀN XỬ LÝ!")
    print(f"Dữ liệu sẵn sàng tại: {SAVE_DIR}")

if __name__ == "__main__":
    main()
