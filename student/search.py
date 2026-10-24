from __future__ import annotations

import numpy as np

from game2048.rules import legal_moves, move_board
from game2048.types import Move
from student.heuristic import evaluate_board


def _search_value(board: np.ndarray, depth: int, heuristic_fn=evaluate_board) -> float:
    """Đánh giá trạng thái bằng shallow search cục bộ."""
    if depth <= 0:
        return float(heuristic_fn(board, 0))

    moves = legal_moves(board)
    if not moves:
        return float("-inf")

    best = float("-inf")
    for move in moves:
        next_board, merge_score, changed = move_board(board, move)
        if not changed:
            continue
        score = heuristic_fn(next_board, merge_score)
        future = _search_value(next_board, depth - 1, heuristic_fn)
        # Chỉ cần 1 bước xem trước để tránh bị nặng quá.
        best = max(best, score + 0.35 * future)
    return best


def shallow_search_best_move(
    board: np.ndarray,
    legal: tuple[Move, ...],
    depth: int = 2,
    heuristic_fn=evaluate_board,
) -> Move:
    """Chọn move tốt nhất bằng search cạn, không quá nặng về thời gian."""
    if not legal:
        raise ValueError("Không có move hợp lệ để xét")

    best_move = legal[0]
    best_value = float("-inf")

    for move in legal:
        next_board, merge_score, changed = move_board(board, move)
        if not changed:
            continue

        immediate = heuristic_fn(next_board, merge_score)
        future_value = _search_value(next_board, depth - 1, heuristic_fn)
        total = immediate + 0.4 * future_value

        if total > best_value:
            best_value = total
            best_move = move

    return best_move
