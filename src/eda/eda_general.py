import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class BehanceEDA:
    def __init__(self, interactions_df, item_features_df=None):
        self.df = interactions_df
        self.features = item_features_df
        
    def basic_statistics(self):
        """1. Thống kê cơ bản: Users, Items, Interactions, Density"""
        n_users = self.df['user_id'].nunique()
        n_items = self.df['item_id'].nunique()
        n_inter = len(self.df)
        
        # Công thức tính Density: Số tương tác / (Số user * Số item)
        density = n_inter / (n_users * n_items)
        
        stats = {
            "Total Users": n_users,
            "Total Items": n_items,
            "Total Interactions": n_inter,
            "Density": f"{density:.5%}"
        }
        return pd.Series(stats)

    def plot_interaction_distribution(self):
        """2. Phân phối interactions: Để quyết định ngưỡng filter"""
        user_counts = self.df.groupby('user_id').size()
        item_counts = self.df.groupby('item_id').size()
        
        # Vẽ biểu đồ phân phối (Long-tail distribution)
        # Giúp bạn xác định xem có nên bỏ user < 5 interactions không...
        pass

    def check_sparsity(self):
        """3. Sparsity: Xác định mức độ thưa thớt của action space"""
        pass

    def analyze_image_features(self):
        """4. Image features: Độ bao phủ (Coverage) và phương sai (Variance)"""
        if self.features is not None:
            # Phân tích xem các vector đặc trưng có bị trùng lặp hay quá giống nhau không
            pass
        else:
            print("Chưa có dữ liệu Image Features.")

    def run_all(self):
        print("--- THỐNG KÊ CƠ BẢN ---")
        print(self.basic_statistics())
        # Gọi các hàm vẽ biểu đồ...