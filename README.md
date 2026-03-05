# image-recommendation-rl
## Project Structure
image-recommendation-rl/
│
├── data/ # Dữ liệu chung
│ ├── raw/ # Chứa link Google Drive (không upload file lớn)
│ └── processed/ # Dữ liệu sau tiền xử lý
│
├── eda/ # Phân tích dữ liệu (EDA)
│ ├── eda_common.ipynb # EDA chung
│ ├── eda_dqn.ipynb # EDA cho mô hình DQN (Phú / Long)
│ └── eda_a2c.ipynb # EDA cho mô hình A2C (Oanh)
│
├── preprocessing/ # Tiền xử lý dữ liệu (Oanh)
│ └── preprocess.py # 7 bước tiền xử lý chung
│
├── environment/ # Môi trường RL (Phú / Long)
│ ├── env.py # Environment simulator
│ ├── state.py # Định nghĩa State
│ ├── action.py # Định nghĩa Action
│ └── reward.py # Định nghĩa Reward
│
├── models/ # Các mô hình
│ ├── dqn/ # Mô hình DQN (Phú / Long)
│ │ ├── dqn.py # Q-Network
│ │ ├── replay_buffer.py # Experience Replay
│ │ └── train_dqn.py # Script train DQN
│ │
│ └── a2c/ # Mô hình A2C (Oanh)
│ ├── actor.py # Actor network
│ ├── critic.py # Critic network
│ └── train_a2c.py # Script train A2C
│
├── evaluation/ # Đánh giá mô hình (Oanh)
│ ├── metrics.py # Hit Rate@K, NDCG@K
│ ├── evaluate.py # So sánh 2 mô hình
│ └── results/ # Kết quả thực nghiệm
│ ├── dqn_results.json
│ ├── a2c_results.json
│ └── comparison.png # Biểu đồ so sánh
│
├── docs/ # Tài liệu dự án
│ ├── plan.md # Kế hoạch thực hiện
│ ├── dataset.md # Mô tả dataset Behance
│ ├── model.md # Giải thích DQN và A2C
│ └── results.md # Tổng hợp kết quả
│
├── app/ # Giao diện demo
│ ├── static/
│ ├── templates/
│ └── app.py
│
├── .gitignore
├── requirements.txt
└── README.md
