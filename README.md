## CẤU TRÚC THƯ MỤC DỰ ÁN

```
image-recommendation-rl/
├── data/
├── eda/
├── preprocessing/
├── environment/
├── models/
├── evaluation/
├── docs/
├── app/
├── .gitignore
├── requirements.txt
└── README.md
```

---

### Mô tả chi tiết

**`data/`** — Dữ liệu Behance
- `raw/` — Chứa link Google Drive đến file gốc (không upload file lớn lên GitHub)
- `processed/` — Dữ liệu sau khi tiền xử lý, sẵn sàng đưa vào train

**`eda/`** — Phân tích khám phá dữ liệu
- `eda_common.ipynb` — Phân tích chung cho cả 2 mô hình
- `eda_dqn.ipynb` — Phân tích riêng phục vụ DQN
- `eda_a2c.ipynb` — Phân tích riêng phục vụ A2C

**`preprocessing/`** — Tiền xử lý dữ liệu
- `preprocess.py` — Pipeline tiền xử lý chung 7 bước

**`environment/`** — Môi trường học tăng cường
- `env.py` — Environment simulator
- `state.py` — Định nghĩa State
- `action.py` — Định nghĩa Action
- `reward.py` — Định nghĩa Reward

**`models/`** — Implement 2 thuật toán RL
- `dqn/` — Thuật toán DQN
  - `dqn.py` — Q-network
  - `replay_buffer.py` — Experience Replay Buffer
  - `train_dqn.py` — Script train DQN
- `a2c/` — Thuật toán A2C
  - `actor.py` — Actor network
  - `critic.py` — Critic network
  - `train_a2c.py` — Script train A2C

**`evaluation/`** — Đánh giá và so sánh
- `metrics.py` — Tính Hit Rate@K và NDCG@K
- `evaluate.py` — So sánh kết quả 2 mô hình
- `results/` — Lưu kết quả thực nghiệm
  - `dqn_results.json` — Kết quả DQN
  - `a2c_results.json` — Kết quả A2C
  - `comparison.png` — Biểu đồ so sánh

**`docs/`** — Tài liệu dự án
- `plan.md` — Kế hoạch thực hiện
- `dataset.md` — Mô tả dataset Behance
- `model.md` — Giải thích DQN và A2C
- `results.md` — Tổng hợp kết quả thực nghiệm

**`app/`** — Giao diện demo
- `static/` — CSS, JS, ảnh tĩnh
- `templates/` — HTML templates
- `app.py` — Backend server

**`.gitignore`** — Danh sách file không upload lên GitHub (data lớn, cache...)

**`requirements.txt`** — Danh sách thư viện Python cần cài đặt

**`README.md`** — Mô tả tổng quan dự án
