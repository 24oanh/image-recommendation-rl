import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from eda_general import BehanceEDA # Đảm bảo file eda_general.py nằm cùng thư mục

class BehanceDQNEda(BehanceEDA):
    def __init__(self, interactions_df, item_features_df=None):
        # Kế thừa init từ lớp cha
        super().__init__(interactions_df, item_features_df)
        self.interactions_df = interactions_df
        self.item_features = item_features_df

    def plot_interaction_distribution(self):
        """2. Phân phối interactions: Để quyết định ngưỡng filter (Kế thừa và thực thi)"""
        user_counts = self.df.groupby('user_id').size()
        item_counts = self.df.groupby('item_id').size()

        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        
        sns.histplot(user_counts, bins=50, kde=True, ax=ax[0], log_scale=True)
        ax[0].set_title('Phân phối số lượng tương tác/User (Log Scale)')
        ax[0].set_xlabel('Số lượng Interactions')

        sns.histplot(item_counts, bins=50, kde=True, ax=ax[1], log_scale=True)
        ax[1].set_title('Phân phối Popularity của Items (Log Scale)')
        ax[1].set_xlabel('Số lượng Interactions')
        
        plt.tight_layout()
        plt.show()

    def analyze_action_space(self, filter_threshold=5):
        """Phân tích kích thước action space sau filter để quyết định candidate pool"""
        # Giả định filter các user có quá ít tương tác
        user_counts = self.df.groupby('user_id').size()
        active_users = user_counts[user_counts >= filter_threshold].index
        filtered_df = self.df[self.df['user_id'].isin(active_users)]
        
        n_items_filtered = filtered_df['item_id'].nunique()
        print(f"--- PHÂN TÍCH ACTION SPACE (Threshold >= {filter_threshold}) ---")
        print(f"Số lượng Items còn lại (Potential Actions): {n_items_filtered}")
        
        # Phân tích Long-tail để chọn Top-K candidate
        item_pop = filtered_df.groupby('item_id').size().sort_values(ascending=False)
        top_10_percent = int(len(item_pop) * 0.1)
        coverage_top_10 = item_pop.iloc[:top_10_percent].sum() / len(filtered_df)
        
        print(f"Top 10% items chiếm {coverage_top_10:.2%} tổng tương tác.")
        return n_items_filtered

    def analyze_reward_sparsity(self):
        """Phân tích reward sparsity theo từng user để đánh giá Replay Buffer"""
        # Trong DQN, reward thường dựa trên click/tương tác
        # Nếu một user có 1000 items nhưng chỉ tương tác 5, reward cực sparse.
        user_reward_density = self.df.groupby('user_id')['item_id'].nunique()
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=user_reward_density)
        plt.title('Phân phối Reward Sparsity (Số interaction thực tế mỗi User)')
        plt.show()

    def analyze_image_features(self):
        """Phân tích State Representation (Image Features) qua PCA"""
        if self.item_features is not None:
            print("--- PHÂN TÍCH STATE REPRESENTATION ---")
            # Kiểm tra phương sai bằng PCA để xem vector có đủ 'giàu' thông tin không
            features_matrix = self.item_features.drop(columns=['item_id']).values
            pca = PCA()
            pca.fit(features_matrix)
            
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            n_95 = np.where(cumulative_variance >= 0.95)[0][0] + 1
            
            print(f"Số lượng thành phần chính để đạt 95% variance: {n_95}/{features_matrix.shape[1]}")
            
            plt.plot(cumulative_variance)
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Explained Variance')
            plt.title('PCA Variance Analysis cho Image Features')
            plt.grid(True)
            plt.show()
        else:
            print("Cảnh báo: Không có dữ liệu Image Features để phân tích.")

    def run_dqn_eda(self):
        """Chạy toàn bộ quy trình EDA cho DQN"""
        print(self.basic_statistics())
        self.plot_interaction_distribution()
        self.analyze_action_space()
        self.analyze_reward_sparsity()
        self.analyze_image_features()