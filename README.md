# image-recommendation-rl
image-recommendation-rl/
│
├── data/                          # CHUNG
│   ├── raw/                       # Để link Google Drive (không upload file lớn)
│   └── processed/                 # Output sau tiền xử lý của Oanh
│
├── eda/                           # CHUNG + RIÊNG
│   ├── eda_common.ipynb           # EDA chung
│   ├── eda_dqn.ipynb              # EDA riêng DQN (Phú/Long)
│   └── eda_a2c.ipynb              # EDA riêng A2C (Oanh)
│
├── preprocessing/                 # OANH
│   ├── preprocess.py              # 7 bước tiền xử lý chung
│
├── environment/                   # PHÚ/LONG
│   ├── env.py                     # Environment simulator
│   ├── state.py                   # Định nghĩa State
│   ├── action.py                  # Định nghĩa Action
│   └── reward.py                  # Định nghĩa Reward
│
├── models/                        # RIÊNG
│   ├── dqn/                       # PHÚ/LONG
│   │   ├── dqn.py                 # Q-network
│   │   ├── replay_buffer.py       # Experience Replay
│   │   └── train_dqn.py           # Train script
│   └── a2c/                       # OANH
│       ├── actor.py               # Actor network
│       ├── critic.py              # Critic network
│       └── train_a2c.py           # Train script
│
├── evaluation/                    # OANH
│   ├── metrics.py                 # Hit Rate@K, NDCG@K
│   ├── evaluate.py                # So sánh 2 mô hình
│   └── results/                   # Kết quả thực nghiệm
│       ├── dqn_results.json
│       ├── a2c_results.json
│       └── comparison.png         # Biểu đồ so sánh
│
├── docs/                          # Tài liệu dự án
│   ├── plan.md                    # Kế hoạch thực hiện
│   ├── dataset.md                 # Mô tả dataset Behance
│   ├── model.md                   # Giải thích DQN và A2C
│   └── results.md                 # Tổng hợp kết quả
│
├── app/                           # Giao diện demo
│   ├── static/
│   ├── templates/
│   └── app.py
│
├── .gitignore
├── requirements.txt
└── README.md
