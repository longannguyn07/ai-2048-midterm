"""Adaptive expectimax. Depth counts future player moves after a root move.
Every root move is followed by a spawn. Depth zero evaluates that spawn;
depth one also searches one more player move and its following spawn.
"""
from __future__ import annotations

import time
from functools import lru_cache

BASE = tuple((r, c if r % 2 == 0 else 3 - c) for r in range(4) for c in range(4))

def _paths():
    paths = set()
    for reflect in (False, True):
        for rotation in range(4):
            cells = []
            for row, col in BASE:
                if reflect:
                    col = 3 - col
                for _ in range(rotation):
                    row, col = col, 3 - row
                cells.append(4 * row + col)
            paths.add(tuple(cells))
    return tuple(sorted(paths))

PATHS = _paths()
ROWS = tuple(tuple(4 * r + c for c in range(4)) for r in range(4))
COLS = tuple(tuple(4 * r + c for r in range(4)) for c in range(4))
EDGES = tuple((i, i + 1) for i in range(16) if i % 4 < 3) + tuple((i, i + 4) for i in range(12))
WEIGHTS = tuple(0.77 ** i for i in range(16))
TERMINAL_VALUE = -10_000_000.0
EXACT_EMPTY_LIMIT = 5
APPROX_POSITIONS = 4
MERGE_REWARD_WEIGHT = 1.0
ORIENTATION_HYSTERESIS = 1.18

@lru_cache(maxsize=65536)
def _merge_left(row):
    tiles = [x for x in row if x]
    result, reward, i = [], 0, 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            exponent = tiles[i] + 1
            result.append(exponent)
            reward += 1 << exponent
            i += 2
        else:
            result.append(tiles[i])
            i += 1
    return tuple(result + [0] * (4 - len(result))), reward

@lru_cache(maxsize=120000)
def _move(board, direction):
    result, reward = list(board), 0
    for line in (ROWS if direction >= 2 else COLS):
        positions = line if direction in (0, 2) else line[::-1]
        merged, gain = _merge_left(tuple(board[i] for i in positions))
        for position, value in zip(positions, merged):
            result[position] = value
        reward += gain
    moved = tuple(result)
    return moved, reward, moved != board

def _legal_count(board):
    return sum(_move(board, move)[2] for move in range(4))

def _leaf_value(board, orientation):
    """A full board with no legal merge is a loss, even at the horizon."""
    return TERMINAL_VALUE if 0 not in board and _legal_count(board) == 0 else _evaluate(board, orientation)

def _exact_chance_leaf(board, orientation, leaf=None):
    """Exact one-spawn expectation at the depth-zero horizon."""
    if leaf is None:
        leaf = lambda state: _leaf_value(state, orientation)
    empty = [i for i, tile in enumerate(board) if tile == 0]
    if not empty:
        return leaf(board)
    return sum(0.9 * leaf(board[:i] + (1,) + board[i + 1:])
               + 0.1 * leaf(board[:i] + (2,) + board[i + 1:])
               for i in empty) / len(empty)

@lru_cache(maxsize=120000)
def _evaluate(board, orientation):
    empty = board.count(0)
    path = PATHS[orientation]
    values = [board[i] for i in path]
    phase = 0 if empty >= 8 else (1 if empty >= 4 else 2)
    snake = sum((1 << e) ** 1.1 * WEIGHTS[i] for i, e in enumerate(values) if e)
    inversions = sum(max(0, values[i + 1] - values[i]) * (1 + max(values[i:i + 2]) / 8)
                     for i in range(15))
    smooth = sum(abs(board[a] - board[b]) * (1 + max(board[a], board[b]) / 12)
                 for a, b in EDGES if board[a] and board[b])
    merge = 0
    for line in ROWS + COLS:
        compact = [board[i] for i in line if board[i]]
        merge += sum(min(compact[i], 10) for i in range(len(compact) - 1)
                     if compact[i] == compact[i + 1])
    top = max(board)
    corner = board[path[0]] == top
    top_order = sum(max(0, values[i] - values[i + 1]) * (1 - i / 8) for i in range(4))
    mobility = _legal_count(board) if empty <= 4 else 4
    return (1.3 * snake + (260, 315, 390)[phase] * empty ** 1.7
            - (18, 28, 40)[phase] * inversions
            - (18, 24, 28)[phase] * smooth
            + (45, 65, 80)[phase] * merge
            + (90, 130, 200)[phase] * corner * top
            + 20 * top_order + (0, 50, 180)[phase] * mobility)

def _orientation_score(board, path):
    return sum((1 << board[position]) ** 1.1 * WEIGHTS[i]
               for i, position in enumerate(path) if board[position])

def _quick_score(board, orientation):
    """Cheap emergency move ranking, without search or mobility expansion."""
    path = PATHS[orientation]
    snake = sum((1 << board[position]) * WEIGHTS[i]
                for i, position in enumerate(path) if board[position])
    return snake + 240 * board.count(0) + 25 * board[path[0]]

class _TimedOut(Exception):
    pass

class Agent:
    name = "Adaptive-Expectimax"

    def __init__(self):
        self._orientation = None
        self._cache = {}
        self.last_completed_depth = 0
        self.last_nodes = 0
        self.orientation_switches = 0
        self.total_stats = {}
        self.depth_by_phase = {"early": [], "mid": [], "late": []}
        self._reset_counters()

    def _reset_counters(self):
        self.last_player_nodes = 0
        self.last_chance_nodes = 0
        self.last_heuristic_evals = 0
        self.last_terminal_evals = 0
        self.last_cache_hits = 0
        self.last_cache_misses = 0
        self.last_exact_spawn_children = 0
        self.last_sampled_spawn_children = 0
        self.last_timeouts = 0
        self.last_completed_root = False

    def _record_totals(self, board):
        names = ("player_nodes", "chance_nodes", "heuristic_evals", "terminal_evals",
                 "cache_hits", "cache_misses", "exact_spawn_children",
                 "sampled_spawn_children", "timeouts")
        for name in names:
            self.total_stats[name] = self.total_stats.get(name, 0) + getattr(self, "last_" + name)
        phase = "early" if board.count(0) >= 8 else ("mid" if board.count(0) >= 4 else "late")
        self.depth_by_phase[phase].append(self.last_completed_depth if self.last_completed_root else -1)

    def reset(self, seed=None):
        self._orientation = None
        self._cache.clear()
        self.last_completed_depth = 0
        self.last_nodes = 0
        self.orientation_switches = 0
        self.total_stats = {}
        self.depth_by_phase = {"early": [], "mid": [], "late": []}
        self._reset_counters()

    def _select_orientation(self, board):
        scores = [_orientation_score(board, path) for path in PATHS]
        best = max(range(len(PATHS)), key=scores.__getitem__)
        if self._orientation is None or scores[best] > ORIENTATION_HYSTERESIS * scores[self._orientation]:
            if self._orientation is not None and best != self._orientation:
                self.orientation_switches += 1
            self._orientation = best
        return self._orientation

    def _time_budget(self, obs, limit_ms, board):
        remaining = min(limit_ms, obs.time_remaining_ms if obs.time_remaining_ms is not None else limit_ms)
        if obs.total_time_limit_ms is None:
            return min(remaining * 0.7, 25.0)
        projected = max(300, 1000 - obs.move_count)
        phase = 0.3 if board.count(0) >= 8 else (0.6 if board.count(0) >= 4 else 0.9)
        return max(0.1, min(remaining / projected * phase, 220.0, remaining * 0.025))

    def choose_move(self, obs, time_limit_ms):
        legal = tuple(obs.legal_moves)
        if not legal:
            raise ValueError("No legal moves")
        self.last_completed_depth = self.last_nodes = 0
        self._reset_counters()
        if len(legal) == 1:
            return legal[0]
        board = tuple(0 if int(x) == 0 else int(x).bit_length() - 1 for x in obs.board.flat)
        orientation = self._select_orientation(board)
        options = [(move, *_move(board, int(move))[:2]) for move in legal]
        remaining_ms = min(time_limit_ms, obs.time_remaining_ms if obs.time_remaining_ms is not None else time_limit_ms)
        reserve_ms = max(2.0, obs.total_time_limit_ms * 0.01) if obs.total_time_limit_ms is not None else 2.0
        if remaining_ms < reserve_ms:
            best = max(options, key=lambda o: _quick_score(o[1], orientation) + 0.5 * o[2])[0]
            self._record_totals(board)
            return best
        best = max(options, key=lambda o: _evaluate(o[1], orientation) + MERGE_REWARD_WEIGHT * o[2])[0]
        deadline = time.perf_counter() + self._time_budget(obs, time_limit_ms, board) / 1000
        if len(self._cache) > 120000:
            self._cache.clear()

        def tick(player_node):
            self.last_nodes += 1
            if player_node:
                self.last_player_nodes += 1
            else:
                self.last_chance_nodes += 1
            if self.last_nodes % 32 == 0 and time.perf_counter() >= deadline:
                raise _TimedOut

        def leaf(state, fast=False):
            if fast:
                self.last_nodes += 1
            if 0 not in state and _legal_count(state) == 0:
                self.last_terminal_evals += 1
                return TERMINAL_VALUE
            self.last_heuristic_evals += 1
            return _evaluate(state, orientation)

        def player(state, depth):
            tick(True)
            key = (state, depth, 1, orientation)
            if key in self._cache:
                self.last_cache_hits += 1
                return self._cache[key]
            self.last_cache_misses += 1
            if depth == 0:
                return leaf(state)
            moves = [(_move(state, direction)) for direction in range(4)]
            moves = [(out, reward) for out, reward, changed in moves if changed]
            if not moves:
                self.last_terminal_evals += 1
                return TERMINAL_VALUE
            result = max(MERGE_REWARD_WEIGHT * reward + chance(out, depth - 1)
                         for out, reward in moves)
            self._cache[key] = result
            return result

        def chance(state, depth):
            tick(False)
            key = (state, depth, 0, orientation)
            if key in self._cache:
                self.last_cache_hits += 1
                return self._cache[key]
            self.last_cache_misses += 1
            empty = [i for i, value in enumerate(state) if value == 0]
            if not empty:
                return player(state, depth)
            if depth == 0 and len(empty) <= EXACT_EMPTY_LIMIT:
                self.last_exact_spawn_children += 2 * len(empty)
                result = _exact_chance_leaf(state, orientation, lambda spawned: leaf(spawned, True))
                self._cache[key] = result
                return result
            if len(empty) <= EXACT_EMPTY_LIMIT:
                representatives = [(i, 1 / len(empty)) for i in empty]
            else:
                # Early cells in the preferred snake path are structurally
                # dangerous to fill; later cells represent ordinary spawns.
                rank = {position: order for order, position in enumerate(PATHS[orientation])}
                ranked = sorted(empty, key=rank.__getitem__)
                representatives = []
                groups = min(APPROX_POSITIONS, len(empty))
                for group in range(groups):
                    members = ranked[group * len(empty) // groups:(group + 1) * len(empty) // groups]
                    representative = members[0] if group == 0 else members[len(members) // 2]
                    representatives.append((representative, len(members) / len(empty)))
            result = 0.0
            for position, probability in representatives:
                state2 = state[:position] + (1,) + state[position + 1:]
                state4 = state[:position] + (2,) + state[position + 1:]
                value2 = player(state2, depth)
                if len(empty) <= EXACT_EMPTY_LIMIT:
                    self.last_exact_spawn_children += 1
                else:
                    self.last_sampled_spawn_children += 1
                value4 = player(state4, depth)
                if len(empty) <= EXACT_EMPTY_LIMIT:
                    self.last_exact_spawn_children += 1
                else:
                    self.last_sampled_spawn_children += 1
                result += probability * (0.9 * value2 + 0.1 * value4)
            self._cache[key] = result
            return result

        for depth in range(6):
            try:
                ordered = sorted(options, key=lambda o: o[0] != best)
                scores = [(MERGE_REWARD_WEIGHT * reward + chance(out, depth), move)
                          for move, out, reward in ordered]
                best = max(scores, key=lambda item: item[0])[1]
                self.last_completed_depth = depth
                self.last_completed_root = True
            except _TimedOut:
                self.last_timeouts += 1
                break
            if time.perf_counter() >= deadline:
                break
        self._record_totals(board)
        return best
