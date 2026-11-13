"""Agent 2048 - Tối ưu chuẩn xác (Không bóp méo nước đi của AI)."""

from __future__ import annotations

import time
import numpy as np

from game2048.observation import Observation
from game2048.rules import legal_moves, move_board
from game2048.types import Move


# =====================================================================
# 1. HÀM ĐÁNH GIÁ THẾ CỜ (CHUẨN XÁC, KHÔNG PHẠT CỨNG NHẮC)
# =====================================================================

SNAKE_WEIGHTS = np.array([
    [65536.0, 32768.0, 16384.0, 8192.0],
    [512.0,   1024.0,  2048.0,  4096.0],
    [256.0,   128.0,   64.0,    32.0],
    [2.0,     4.0,     8.0,     16.0],
], dtype=np.float64) # ma trận rắn gắn điểm trọng số với vị trí


def evaluate_board(board: np.ndarray) -> float: # Hàm tính điểm
    """Tính điểm chất lượng thế cờ dựa hoàn toàn vào ma trận rắn tự nhiên."""
    snake_score = float(np.sum(board * SNAKE_WEIGHTS))
    # lấy điểm từng ô nhân với điểm ma trận rắn -> ô điểm càng lớn trùng với trọng số lớn điểm càng cao
    # => ngầm định hướng AI dồn tile lớn nhất vào góc ròi giảm dần theo rắn
    
    # Thưởng ô trống vừa phải để dễ xoay xở
    empty_cells = int(np.count_nonzero(board == 0))
    # đếm số ô có điểm = 0 <=> trồng --> bằng 0 vì engine cái thế
    empty_score = empty_cells * 20000.0
    # điếm thưởng cho ô trống -> nhân với trọng số phù hợp

    mono_penalty = 0.0
    smooth_penalty = 0.0
    # biến khởi tạo

    for r in range(4):
        for c in range(3):
            val1 = board[r, c]
            val2 = board[r, c + 1] # xét hàng ngang
            if val1 > 0 and val2 > 0:
                smooth_penalty += abs(np.log2(val1) - np.log2(val2)) 
                # đo độ gồ gề <=> điểm lệch càng lớn thì phạt càng nặng -> giá trị tương đồng dễ hợp nhất
            if val1 < val2 and val2 > 0:
                mono_penalty += (np.log2(val2) - np.log2(val1)) if val1 > 0 else np.log2(val2)
                # đo độ tương đồng <=> có theo quy luật tăng hay giảm không -> theo quy luật dễ hợp nhất


    for c in range(4):
        for r in range(3):
            val1 = board[r, c]
            val2 = board[r + 1, c] # xét hàng dọc
            if val1 > 0 and val2 > 0:
                smooth_penalty += abs(np.log2(val1) - np.log2(val2))
            if val1 < val2 and val2 > 0:
                mono_penalty += (np.log2(val2) - np.log2(val1)) if val1 > 0 else np.log2(val2)

    return snake_score + empty_score - (mono_penalty * 2.0) - (smooth_penalty * 500.0)


# =====================================================================
# 2. CẮT TỈA NƯỚC ĐI
# =====================================================================

def get_promising_moves(board: np.ndarray, max_moves: int = 3): # cắt bỏ nước đi tệ ngay từ đầu để giảm gánh nặng lên expectimax
    candidates = [] # danh sách chứa cặp (điểm ứng viên, nước đi)
    for move in legal_moves(board): # duyệt tất cả hướng hợp lệ
        next_board, reward, changed = move_board(board, move) 
        # next_board -> bàn cờ tiếp, reward -> điểm kiếm đc, change(boolean) -> True nếu bàn cờ thay đổi
        if changed:
            score = evaluate_board(next_board) + reward * 2.0 # tính điểm mới
            candidates.append((score, move)) # thêm điểm và hướng đi

    candidates.sort(key=lambda item: item[0], reverse=True) # sắp xếp theo thứ tự giảm dần
    return [move for _, move in candidates[:max_moves]] # lấy 3 nước đi tốt nhất
# thà đi sâu hơn với 3 nước còn hơn đi nông hơn với 4 nước --> cốt lõi vẫn là thời gian


# =====================================================================
# 3. EXPECTIMAX
# =====================================================================

def expectimax(board: np.ndarray, depth: int, is_player: bool, deadline: float) -> float:
    if time.perf_counter() >= deadline:
        return evaluate_board(board)
    # dừng khi hết giờ

    if depth == 0:
        return evaluate_board(board)
    # dừng khi hết độ sâu

    if is_player: # là người chơi <=> nút max
        best_val = float("-inf") # khởi tạo
        moves = get_promising_moves(board, max_moves=3) # lọc TOP 3
        if not moves:
            return -1.0e9 # gameover phạt rất nặng

        for m in moves:
            next_board, reward, changed = move_board(board, m)
            # next_board -> bàn cờ tiếp, reward -> điểm kiếm đc, change(boolean) -> True nếu bàn cờ thay đổi
            if changed:
                val = expectimax(next_board, depth - 1, False, deadline) + (reward * 2.0) 
                # gọi đệ quy, giảm độ sâu, chuyển luợt, cộng điểm sau khi đi
                if val > best_val:
                    best_val = val # lấy điểm tốt nhất
        return best_val # trả về điểm tốt nhất

    else: # lượt của môi trường <=> nút chance
        empty_positions = np.argwhere(board == 0) # tìm tất cả các ô trống
        num_empty = len(empty_positions) # số ô trống để tính xác suất

        if num_empty == 0:
            return evaluate_board(board) # bàn cờ đầy trả vè heuristic

        eval_positions = empty_positions # duyệt tất cả ô trống
        if num_empty > 4 and depth > 1: 
            indices = np.linspace(0, num_empty - 1, min(3, num_empty), dtype=int)
            # np.linspace -> tạo các số cách đều nhau trong 1 khoảng (star,stop,num) num == số lượng giá trị cần taọ | int ép kiểu
            eval_positions = empty_positions[indices]
            # lấy các vị trí tương ứng của chỉ số vừa tạo

            # determinist đc thể hiện ở đây nó giải thích vì sao cùng 1 seed chạy 10 lần như 1 nhưng không phản ánh đúng tầm quan trọng
            # thay vì xét tất cả trường hợp làm khônhg gian tìm kiếm phinh khó kiểm soát thì chọn giá trị đaị điện random có kiểm xoát
            # min chỉ đề phòng ngừa
            # đây chỉ là xấp sỉ không phản ánh đúng espectimax đầy đủ

        prob = 1.0 / len(eval_positions) # xác xuất chọn 1 vị trí trong E vị trí
        expected_val = 0.0 # khởi tạo tổng điểm kì vọng

        for r, c in eval_positions:
            if time.perf_counter() >= deadline:
                return evaluate_board(board)
            # điều kiện dừng thời gian
            # kiểm tra lại vì 1 nút chance có thể pải duyệt nhiều vị trí pải kiểm tra thời gian trc mỗi vị trí để tránh quá giờ

            b2 = board.copy() # tao bản sao để những lần sau tránh bị ảnh hưởng bởi những lần trước
            b2[r, c] = 2 # gán bằng 2
            val2 = expectimax(b2, depth - 1, True, deadline) # đệ quy tính điểm <+> backtracking
            expected_val += 0.9 * val2 # cộng điểm khi sinh ô 2

            if num_empty <= 3 or depth <= 1:
                b4 = board.copy()
                b4[r, c] = 4
                val4 = expectimax(b4, depth - 1, True, deadline)
                expected_val += 0.1 * val4
            # xét với 4
            # Ô 2 không cần if vì nó chiếm 90% xác suất nên bắt buộc phải mô phỏng chính xác.
            # Ô 4 chỉ chiếm 10% nên ta bọc trong if để chỉ tính toán chi tiết ở các tình huống quan trọng (bàn cờ ít ô trống hoặc tầng đáy)
            # còn lại sẽ xấp xỉ bằng giá trị ô 2 để tiết kiệm thời gian và giảm số lần gọi đệ quy."

            #"Việc xấp xỉ $val4 \approx val2$ dựa trên thực tế là ô 4 chỉ chiếm 10% xác suất nên ảnh hưởng tới điểm kỳ vọng rất nhỏ. 
            # Việc xấp xỉ này giúp giảm một nửa số lần gọi đệ quy, tiết kiệm thời gian tính toán để AI có thể tìm kiếm sâu hơn (tăng độ sâu depth), đem lại nước đi tối ưu hơn tổng thể."
            else:
                expected_val += 0.1 * val2 # không pải 4 thì nhân val2

        return expected_val * prob # trả vè điểm nhân xác xuất ô trống


# =====================================================================
# 4. AGENT CHÍNH
# =====================================================================

class Agent:
    name = "AI3-Nhom_2"

    def __init__(self) -> None:
        self.last_completed_depth = 2
        # depth bằng 2 là hợp lí cho gần như toàn bộ bàn cờ nếu có nhiều hơn 3 ô trống
        # phù hợp hướng logic bth như người - máy - người
        # biến này k dùng để điều khiển logic mà để báo cáo thông số cho mô hình bên ngoài -> để engine ghi log
        # vì nó để báo cáo nên pải chính xác độ sâu vừa tìm biến độ sâu cục bộ có thể là 3 nhưng do hết thời gian chỉ tìm 2
        # phân biệt với biến cục bộ

    def reset(self, seed: int | None = None) -> None:
        self.last_completed_depth = 2
        # đưa agent vè trạng thái ban đầu để tiếp tục cho ván mới

    def choose_move(self, obs: Observation, time_limit_ms: float) -> Move: # hàm đưa ra nước đi
        legal = obs.legal_moves
        # lấy danh sách các nước đi hợp lệ

        if not legal:
            raise ValueError("Không còn nước đi hợp lệ")
        # bào lỗi để hệ thống ngoài tự sử lí
        

        if len(legal) == 1:
            return legal[0]
        # nếu chỉ có 1 nước đi trả vè luôn

        budget_sec = min(0.030, max(0.001, (time_limit_ms * 0.7) / 1000.0))
        # quản lí thời gian tìm kiếm 
        # chỉ dùng 70% khoảng thời gian đc cấp
        # ít nhất là 0.001s và nhiều nhất là 0.03s không đc quá

        deadline = time.perf_counter() + budget_sec
        # cập nhật deadline -> khoảng thời gian buộc kết thúc
        # 3 lí do truy cập / đúng logic tên /dễ đọc

        empty_count = int(np.count_nonzero(obs.board == 0)) # đếm số ô trống để xác định độ sâu phù hợp
        depth = 3 if empty_count <= 3 else 2 # nghĩ sâu hơn nếu có ít hơn 3 ô trống
        self.last_completed_depth = depth # cập nhật depth

        best_move = legal[0]
        best_val = float("-inf")
        # khởi tạo nới đi đầu và điểm

        for m in legal:
            nb, reward, changed = move_board(obs.board, m)
            if changed:
                val = expectimax(nb, depth, False, deadline) + (reward * 2.0)
                if val > best_val:
                    best_val = val
                    best_move = m
        # đánh giá và mô phỏng các nước đi

        return best_move

''' Tư duy vận hành thuật toán :
        1/ Nhận trạng thái bàn cờ --> Nhận bàn cờ, nước đi hợp lệ, thời gian đc cấp
        2/ Xác định nước đi có thể thực hiện --> Kiểm tra danh sách nước đi. Nếu k còn thì báo lỗi nếu chỉ 1 => Trả vè
        3/ Thiết lập tài nguyên tìm kiếm --> Tính ngân sách thời gian, xác định deadline và lựa chọn độ sâu tìm phù hợp dựa trên số ô trống
                                                (<= 3 thì 3 , > 3 thì 2)
        4/ Mô phỏng hướng đi --> Lần lượt thử các hướng di chuyển tạo bàn cờ mới và lấy điểm từ việc gộp
        5/ Sử dụng expectimax để dự đoán tương lai :
            1/ Nút Max(người chơi) --> Chọn nút có giá trị cao nhất : V = max(Vi)
            2/ Nút CHANCE(môi trường) : Mô phỏng sinh ô 2 và 4 theo xác xuất (dùng 1 số chick để xấp sỉ)
            3/ Tiếp tục tìm kiếm đệ quy đến hết độ sâu hoặc thời gian
        6/ Đánh giá và so sánh --> Kết hợp giá trị expectimax với điểm thưởng gộp ô để tính điểm cho từng nước đi
        7/ Đưa ra quyết định --> Chọn nước đi có tổng điểm cao nhất và trả về
    
    4 Heuristic :
        1/ Snake --> Ô xếp giảm dần theo hình rắn từ góc
        2/ Empty tiles --> Số ô trống trên bàn cờ
        3/ Smoothness --> Chênh lệch điểm giữa các ô kề nhau
        4/ Monotonicity --> Hàng cột không tăng hoặc giảm đều

    Cơ chế vần hành Expectimax:
        1/ Base case : Vì nó là hàm đệ quy nên buộc pải có điều kiện dừng
                        1/ GameOver
                        2/ Hết thời gian cho phép
                        3/ Hết độ sâu tìm kiếm
        2/ Nếu k còn nước đi nào hợp lệ(GameOver) --> Ném lỗi
           Nếu chỉ có duy nhất 1 nước đi hợp lệ thì trả về luôn
        3/ Cắt tỉa bớt nhánh tìm kiếm bằng cách lấy cái ô làm đại diện tránh phình cây tìm kiếm quá lớn 
            !*! Đây là 1 phép xấp sỉ k phản ánh đúng bản chất bàn cờ
        3/ 2 chế độ :
            1/ Max(người) --> V = Max(Vi)
                - Tính điểm trạng thái tất cả các hướng có thể sau cắt giảm
                    => Lấy điểm của trạng thái tốt nhất
            2/ Chance(môi trường) --> V = sum(1->n)(0,9/E * score(2) + 0.1/E * score(4))
                - Nếu trường hợp sắp thua hoặc sắp hết độ sâu thì mới mô phỏng cả 2 và 4 còn lại mô phỏng hết số 2 kể cả thay số 4
                    +> Vì mô phỏng số 2 và 4 sẽ tăng số nhánh tím kiếm --> Giảm hiệu năng
                    +> Xác xuất ra số 4 là nhỏ hơn rất nhiều so với số 2 --> Nếu bàn cờ còn nhiều ô trống ==> Ít quan trọng
                    !*! Đây là 1 phép sấp sỉ k phản ánh đúng bản chất bàn cờ ->> Tốt cho hiệu năng
                - Tất cả các chỗ tính điểm đều là cộng bằng mới đúng công thức
                - Lấy điểm nhân với xác xuất ô trống
'''
