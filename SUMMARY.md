# Tổng hợp kiến thức và tiến độ làm bài 2048

## 1. Mục tiêu của bài

Mục tiêu là hoàn thiện file `student/agent.py` để tạo ra một agent chơi 2048 hợp lệ, mạnh hơn baseline, và chạy được trong benchmark của project.

Yêu cầu cần nhớ:
- giữ nguyên tên class `Agent`
- có `reset(seed=None)`
- có `choose_move(obs, time_limit_ms)`
- trả về một move hợp lệ trong `obs.legal_moves`
- không sửa `obs.board`
- không vượt tổng thời gian budget
- không lỗi runtime

---

## 2. API và dữ liệu quan trọng

### `Observation`
Dữ liệu chính trong `Observation`:
- `board`: ma trận NumPy 4x4
- `score`: điểm hiện tại
- `move_count`: số bước đã đi
- `legal_moves`: tuple các nước hợp lệ
- `time_remaining_ms`: thời gian còn lại
- `total_time_limit_ms`: tổng budget

### `Move`
Các hướng đi: `UP`, `DOWN`, `LEFT`, `RIGHT`.

### Logic chọn nước
Agent phải chọn 1 move nằm trong `obs.legal_moves`.

---

## 3. File cốt lõi cần đọc

- `student/agent.py`: file chính cần sửa
- `src/game2048/observation.py`: mô tả dữ liệu đầu vào
- `src/game2048/rules.py`: contains rule of 2048, `move_board`, `legal_moves`
- `src/game2048/benchmark.py`: benchmark và tổng hợp kết quả
- `src/game2048/metrics.py`: summary metrics cho benchmark
- `README.md`: quy định và hướng dẫn bài nộp
- `student/examples/random_agent.py`: ví dụ agent rất đơn giản

---

## 4. Ý tưởng về heuristic

Heuristic là hàm chấm giá trị bàn cờ.

Một heuristic mạnh hơn không chỉ nhìn:
- số ô trống
- tile lớn nhất
- merge_score

Mà còn cần dựa vào:
- smoothness: bàn cờ càng mượt càng tốt
- monotonicity: hàng/cột có xu hướng sắp xếp gọn
- corner bias: giữ tile lớn ở góc
- merge potential: ưu tiên khả năng gộp tiếp theo
- snake pattern: xếp các tile theo dạng ổn định

### Tại sao `p_2048` và `p_4096` thấp là dấu hiệu cần tối ưu heuristic?
Vì khi agent không đạt được tile lớn thường xuyên, nghĩa là heuristic đang đánh giá trạng thái không hợp lý.

---

## 5. Khái niệm shallow search

Heuristic alone chỉ xét trạng thái hiện tại.

Shallow search giúp xét thêm 1 bước tiếp theo:
- thử mỗi move
- simulate bước đi tiếp theo
- đánh giá trạng thái mới
- chọn move có tổng giá trị tốt nhất

Điều này giúp tránh:
- chọn move “trông mạnh ngay lúc đó” nhưng phá hỏng bàn cờ về lâu dài
- chọn move ngắn hạn nhưng không tạo điều kiện merge tốt

---

## 6. Cấu trúc file hỗ trợ được tạo

### `student/heuristic.py`
Tạo các hàm:
- `evaluate_board(board, merge_score=0)`
- `choose_best_move(board, legal, max_depth=2)`

Các yếu tố chính:
- empty cells
- max tile
- corner value
- smoothness
- monotonicity
- merge potential
- snake pattern

### `student/search.py`
Tạo hàm:
- `_search_value(...)`
- `shallow_search_best_move(...)`

Mục tiêu:
- dự đoán giá trị của bàn cờ sau 1-2 bước tiếp theo
- nâng cao chất lượng chọn move mà không quá nặng thời gian

---

## 7. Tích hợp vào `student/agent.py`

`Agent` được thiết kế để:
- có `name = "heuristic-2048"`
- có `reset(seed=None)`
- có `choose_move(obs, time_limit_ms)`
- sử dụng `evaluate_board` từ `student.heuristic`
- sử dụng `shallow_search_best_move` từ `student.search`

Hướng tích hợp:
- `choose_move` lấy danh sách `obs.legal_moves`
- chấm từng move bằng `shallow_search_best_move(obs.board, obs.legal_moves, depth=2, heuristic_fn=evaluate_board)`
- trả về move tốt nhất

---

## 8. Cách đọc benchmark

Benchmark xuất các chỉ số chính:
- `games`: số ván chạy
- `mean_score`: điểm trung bình
- `median_score`: điểm trung vị
- `mean_max_tile`: tile lớn nhất trung bình
- `p_2048`: tỷ lệ đạt 2048
- `p_4096`: tỷ lệ đạt 4096
- `mean_moves`: số nước trung bình
- `mean_decision_time_ms`: thời gian suy nghĩ trung bình mỗi nước
- `timeouts`: số lần vượt thời gian
- `illegal_moves`: số nước sai luật
- `exceptions`: số lỗi runtime
- `budget_exhausted`: số ván vượt tổng ngân sách

### Nếu `p_2048` và `p_4096` thấp thì nên làm gì?
Nên tối ưu heuristic trước, vì điều đó phản ánh khả năng agent tạo tile lớn và duy trì bàn cờ tốt trong suốt ván.

---

## 9. Cách kiểm tra trước khi nộp

Chạy benchmark theo hướng dẫn:

```powershell
python scripts/benchmark_agent.py --agent student.agent:Agent --seeds 1 --start-seed 42 --total-time-sec 5
```

Và nếu cần:

```powershell
python scripts/benchmark_agent.py --agent student.agent:Agent --seeds 5 --start-seed 1000 --total-time-sec 60 --output results/student_validation.json
```

Kiểm tra:
- không exception
- không illegal_moves
- không timeouts nếu không cần
- không budget_exhausted
- điểm tốt hơn baseline

---

## 10. Lộ trình hiệu quả để cải thiện agent

### Bước 1: hiểu và giữ đúng API
- `Agent` phải đúng format
- `choose_move` phải luôn trả legal move

### Bước 2: fix heuristic
- cải thiện smoothness, monotonicity, corner bias, merge potential

### Bước 3: thêm shallow search nhẹ
- không quá sâu để tránh tốn thời gian
- depth 1–2 rất hợp lý

### Bước 4: benchmark nhiều seed
- đánh giá trên nhiều ván để giảm lệch do may mắn

### Bước 5: tune trọng số và độ sâu
- nếu p_2048 thấp, tăng độ ưu tiên cho corner/monotonic
- nếu chậm, giảm depth hoặc tối giản tính toán

---

## 11. Kết luận

Bài toán này không đòi hỏi viết thuật toán cực mạnh như AI chuyên sâu, nhưng cần hiểu rõ 3 phần chính:
1. luật 2048
2. cách đánh giá bàn cờ bằng heuristic
3. cách chọn nước hợp lý bằng shallow search

Dựa trên đó, file `student/agent.py` có thể cải thiện đáng kể so với baseline starter.

---

## 12. Tái bút

Trong quá trình làm, ưu tiên:
- đúng luật
- không error
- agent ổn định
- điểm càng cao càng tốt

Cách làm hiệu quả nhất là:
- heuristic tốt + shallow search nhẹ + benchmark đều đặn
