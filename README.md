# Behance Image Recommendation with Deep Reinforcement Learning (DQN vs A2C)

Dự án Xây dựng hệ thống gợi ý ảnh nghệ thuật dựa trên hành vi người dùng sử dụng Học tăng cường.

## Cấu trúc dự án
Dự án được thiết kế theo chuẩn Pipeline Kỹ thuật phần mềm:
- `data/`: Chứa dữ liệu thô và dữ liệu đã qua tiền xử lý.
- `training_pipeline/`:
  - `models/`: Định nghĩa kiến trúc Mạng Neural (Transformer, DQN, A2C).
  - `environment/`: Định nghĩa Môi trường tương tác MDP (State, Action, Reward).
  - `01_preprocessing.py`: tiền xử lý (K-Core, PCA, Train/Test split).
  - `02_train_dqn.py`: Script huấn luyện mô hình DQN (Baseline).
  - `03_train_a2c.py`: Script huấn luyện mô hình A2C (Proposed).
  - `04_export_demo.py`: Tự động đóng gói trọng số để đẩy sang UI.
  - `05_evaluate.py`: Chạy đánh giá (Hit Rate, NDCG, Coverage).
- `zeppelin_demo/`: Chứa file JSON giao diện chạy trên Apache Zeppelin.

## Cách chạy dự án

1. Cài đặt thư viện: `pip install -r requirements.txt`
2. Đặt 2 file data gốc (`Behance_appreciate_1M.txt`, `Behance_Image_Features.b`, `Behance_Item_to_Owners.gz`) vào thư mục `data/raw/`
3. Chạy toàn bộ Pipeline theo thứ tự sau:
   ```bash
   python training_pipeline/01_preprocessing.py
   python training_pipeline/02_train_dqn.py
   python training_pipeline/03_train_a2c.py
   python training_pipeline/05_evaluate.py
   python training_pipeline/04_export_demo.py
