# Kế Hoạch Thực Hiện Dự Án
# Hệ Thống Gợi Ý Nội Dung Hình Ảnh Trên Mạng Xã Hội Sử Dụng Học Tăng Cường

---

## Tổng Quan Dự Án

| Thông tin | Chi tiết |
|---|---|
| **Dataset** | Behance (McAuley UCSD) — 1M interactions, ảnh nghệ thuật mạng xã hội |
| **Bài toán** | Gợi ý ảnh phù hợp cho từng user dựa trên lịch sử appreciate |
| **Thuật toán** | DQN (baseline) vs A2C (đề xuất) |
| **Metric chính** | Hit Rate@K, NDCG@K |
| **Metric phụ** | Cumulative Reward, Convergence Speed |

---

## Phân Công

| Thành viên | Phụ trách |
|---|---|
| **Oanh** | Giai đoạn 2 (Tiền xử lý), Giai đoạn 4 (A2C), Giai đoạn 5 (Đánh giá) |
| **Phú/Long** | Giai đoạn 3 (Môi trường RL), Giai đoạn 4 (DQN) |
| **Cả hai** | Giai đoạn 1 (EDA chung), viết báo cáo |

---

## Giai Đoạn 1 — Phân Tích Dữ Liệu (EDA)

### Mục tiêu
Hiểu rõ đặc điểm của dataset Behance trước khi xử lý, từ đó đưa ra các quyết định có căn cứ cho các bước tiếp theo.

### 1.1 EDA Chung *(Cả hai — bắt buộc)*

| Bước | Nội dung | Mục đích |
|---|---|---|
| Thống kê cơ bản | Số users, items, interactions, density | Biết quy mô thực tế của dataset |
| Phân phối interactions | Phân phối số interactions mỗi user/item | Quyết định ngưỡng filter phù hợp |
| Sparsity | Density sau filter, action space thực tế | Xác định mức độ sparse để chọn ngưỡng filter |
| Image features | Coverage và variance của feature vectors | Đảm bảo State representation có chất lượng |

> **Tùy chọn:** Phân tích temporal (phân bố theo thời gian), phân tích sau khi filter.

### 1.2 EDA Riêng DQN *(Phú/Long)*

| Nội dung | Mục đích |
|---|---|
| Phân tích kích thước action space sau filter | Quyết định kích thước candidate pool cho DQN |
| Phân tích phân phối popularity của items | Xây dựng candidate pool top-K hợp lý |
| Phân tích reward sparsity theo từng user | Đánh giá chất lượng Replay Buffer |

### 1.3 EDA Riêng A2C *(Oanh)*

| Nội dung | Mục đích |
|---|---|
| Phân tích độ dài trung bình episode | Đảm bảo episode đủ dài để A2C học được |
| Phân tích variance của reward | Điều chỉnh entropy coefficient phù hợp |
| Phân tích sequential pattern của user | Xác nhận A2C phù hợp với behavior của users |

### Ưu tiên thực hiện EDA
```
Ưu tiên cao (bắt buộc):
1. Sparsity + Action space size
2. Image features coverage
3. Phân phối interactions

Ưu tiên thấp (tùy chọn):
4. Temporal analysis
5. Phân tích sau filter
```

---

## Giai Đoạn 2 — Tiền Xử Lý Dữ Liệu *(Oanh)*

### Mục tiêu
Chuẩn bị dataset sạch, chuẩn hóa và có cấu trúc phù hợp để đưa vào train 2 mô hình.

### 2.1 Tiền Xử Lý Chung *(7 bước — output dùng chung cho cả DQN và A2C)*

| Bước | Công việc | Ý nghĩa |
|---|---|---|
| 1 | Load & đọc dữ liệu | Giải nén file `.gz`, `.b` thành cấu trúc có thể xử lý |
| 2 | Filter 5-core | Loại bỏ users/items có ít hơn 5 interactions để tăng reward density |
| 3 | Map ID về index liên tục | Chuyển ID dạng chuỗi về số nguyên 0, 1, 2... để model xử lý được |
| 4 | Chia train/test theo timestamp | 80% đầu làm train, 20% cuối làm test — tránh data leakage |
| 5 | Xử lý image features | Giảm chiều 4096 → 256 bằng PCA, chuẩn hóa về cùng scale |
| 6 | Xây dựng user history | Sắp xếp interactions theo timestamp, tổ chức thành chuỗi lịch sử mỗi user |
| 7 | Xây dựng candidate pool | Lấy top-1000 items phổ biến nhất làm action space |

### 2.2 Tiền Xử Lý Riêng

| Mô hình | Công việc | Ý nghĩa |
|---|---|---|
| **DQN** | Tinh chỉnh candidate pool | Đảm bảo kích thước phù hợp với Q-network |
| **A2C** | Tinh chỉnh user history trajectory | Đảm bảo trajectory đủ dài cho Advantage estimation |

### Output của Giai đoạn 2
```
data/processed/
├── interactions_train.pkl     # Tập train
├── interactions_test.pkl      # Tập test
├── user_history.pkl           # Lịch sử tương tác mỗi user
├── image_features_256.pkl     # Feature vectors sau PCA
├── candidate_pool.pkl         # Top-1000 items
└── id_mappings.pkl            # Mapping user_id, item_id → index
```

---

## Giai Đoạn 3 — Xây Dựng Môi Trường RL *(Phú/Long)*

### Mục tiêu
Xây dựng môi trường giả lập (simulator) từ offline log Behance để cả DQN và A2C có thể tương tác và học.

### Các thành phần

| Thành phần | Định nghĩa | Chi tiết |
|---|---|---|
| **State** | Agent đang "thấy gì" | Mean pooling K=5 image feature vectors của K items user appreciate gần nhất → vector 256-dim |
| **Action** | Agent có thể "làm gì" | Chọn 1 item trong candidate pool 1000 items |
| **Reward** | Tín hiệu phản hồi | +1 nếu user appreciate item được gợi ý, 0 nếu không |
| **Environment Simulator** | Giả lập môi trường | Đọc interaction log → trả về (next_state, reward, done) sau mỗi action |

### Cấu trúc Environment
```python
env = BehanceEnvironment(data)

state = env.reset(user_id)        # Khởi tạo state ban đầu của user
next_state, reward, done = env.step(action)  # Thực hiện action, nhận reward
```

### Ưu tiên thực hiện
```
1. Environment Simulator  → nền tảng của toàn bộ thực nghiệm
2. Reward                 → tín hiệu học duy nhất của agent
3. State                  → đầu vào của cả 2 mô hình
4. Action Space           → đơn giản nhất, index candidate pool
```

---

## Giai Đoạn 4 — Implement Mô Hình

### 4.1 DQN *(Phú/Long)*

**Ý nghĩa:** Thuật toán value-based — học giá trị Q(s,a) cho mỗi cặp state-action, chọn action có Q cao nhất.

| Thành phần | Ý nghĩa | Ưu tiên |
|---|---|---|
| **Q-network** | Mạng neural nhận State → output Q-value cho mỗi action | 🔴 Cao nhất |
| **Replay Buffer** | Lưu (state, action, reward, next_state) → lấy mẫu ngẫu nhiên để train | 🔴 Cao |
| **Target Network** | Bản sao Q-network cập nhật chậm → train ổn định | 🔴 Cao |
| **ε-greedy** | Xác suất ε chọn ngẫu nhiên → cân bằng khám phá và khai thác | 🟡 Trung bình |
| **TD loss** | (r + γ maxQ' - Q)² → cập nhật Q-network | 🟡 Trung bình |

### 4.2 A2C *(Oanh)*

**Ý nghĩa:** Thuật toán policy-based — học trực tiếp policy π(a|s) thông qua Actor, đánh giá state qua Critic.

| Thành phần | Ý nghĩa | Ưu tiên |
|---|---|---|
| **Actor network** | Nhận State → output xác suất chọn mỗi action | 🔴 Cao nhất |
| **Critic network** | Nhận State → output V(s) đánh giá state | 🔴 Cao nhất |
| **Advantage** | A = R + γV(s') - V(s) → tín hiệu cập nhật Actor | 🔴 Cao |
| **Actor + Critic loss** | Cập nhật đồng thời policy và value function | 🔴 Cao |
| **Entropy regularization** | Khuyến khích khám phá, tránh collapse policy | 🟡 Trung bình |

---

## Giai Đoạn 5 — Thực Nghiệm & Đánh Giá *(Oanh)*

### Mục tiêu
So sánh hiệu suất DQN và A2C trên cùng dataset, cùng điều kiện thực nghiệm.

### Metric đánh giá

| Metric | Ý nghĩa | Ưu tiên |
|---|---|---|
| **Hit Rate@K** | Tỷ lệ user được gợi ý đúng ít nhất 1 item trong top-K | 🔴 Chính |
| **NDCG@K** | Chất lượng ranking — item đúng xếp càng cao điểm càng cao | 🔴 Chính |
| Cumulative Reward | Tổng reward tích lũy qua quá trình train | 🟡 Phụ |
| Convergence Speed | Số episodes để reward ổn định | 🟡 Phụ |

### Baseline so sánh

| Mô hình | Mô tả |
|---|---|
| Random | Gợi ý ngẫu nhiên — baseline thấp nhất |
| Popularity-based | Luôn gợi ý items phổ biến nhất |
| **DQN** | Value-based RL |
| **A2C** | Policy-based RL |

### Dự kiến kết quả

| Metric | Random | Popularity | DQN | A2C |
|---|---|---|---|---|
| Hit Rate@10 | ~0.05–0.10 | ~0.25–0.35 | ~0.35–0.45 | ~0.45–0.55 |
| NDCG@10 | ~0.02–0.05 | ~0.10–0.18 | ~0.15–0.25 | ~0.20–0.30 |

> **Lưu ý:** Đây là dự kiến — kết quả thực tế phụ thuộc vào hyperparameter tuning.

---

## Kế Hoạch Thời Gian

| Giai đoạn | Công việc | Thời gian |
|---|---|---|
| 1 | EDA | 2 ngày |
| 2 | Tiền xử lý | 2 ngày |
| 3 | Xây dựng môi trường RL | 2 ngày |
| 4 | Implement DQN + A2C | 6 ngày |
| 5 | Thực nghiệm + Đánh giá | 2 ngày |
| — | Viết báo cáo | 4 ngày |
| **Tổng** | | **~18 ngày** |

---

## Hạn Chế Của Dự Án

| Hạn chế | Giải thích |
|---|---|
| Reward binary (0/1) | Behance chỉ có appreciate/không — không có signal đa mức |
| Offline evaluation | Train/test trên log tĩnh, không có real-time feedback |
| Action space giới hạn | Candidate pool 1000 items — không phải toàn bộ dataset |
| Cold start | User/item mới không có interaction history |
| DQN bất lợi action space | DQN tự nhiên kém hiệu quả hơn A2C với large action space |
