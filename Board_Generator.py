import math

import numpy as np


class BoardGenerator:
    def __init__(
            self,
            board_size=12,
            number_of_boards=10000,
            divisions=100,
            max_tag=True,
            seed=None
    ):
        if board_size < 1:
            raise ValueError("board_size must be positive.")

        if divisions < 1:
            raise ValueError("divisions must be positive.")

        if number_of_boards < 1:
            raise ValueError("number_of_boards must be positive.")

        self.count = 0
        self.bs = board_size
        self.range = board_size ** 2
        self.num = number_of_boards
        self.max_tag = max_tag

        if self.range + 1 < divisions:
            print(
                f"\033[91mWARNING: Requested {divisions} divisions, "
                f"but a {board_size}×{board_size} board supports "
                f"only {self.range} divisions. "
                f"Capping divisions at {self.range}.\033[0m"
            )
            divisions = self.range

        elif self.range % divisions != 0:
            print(
                "\033[91mWARNING: divisions do not evenly divide "
                "the total number of board cells. "
                "Some occupancy levels will be rounded or skipped.\033[0m"
            )
        self.div = divisions
        self.rng = np.random.default_rng(seed)

    def generate_unique(self, n, l, low=0, high=100):
        total = high - low

        if n > math.comb(total, l):
            raise ValueError("Requested more unique boards than possible.")

        if l == 0:
            return np.empty((n, 0), dtype=np.int32)

        w = (total + 7) // 8
        packed = np.empty((0, w), dtype=np.uint8)

        while len(packed) < n:
            missing = n - len(packed)
            m = min(missing + missing // 20 + 16, 100_000)

            r = self.rng.random((m, total), dtype=np.float32)
            idx = np.argpartition(r, l - 1, axis=1)[:, :l]
            b = np.zeros((m, total), dtype=np.uint8)
            np.put_along_axis(b, idx, 1, axis=1)

            packed = np.concatenate([packed, np.packbits(b, axis=1)])

            v = packed.view(np.dtype((np.void, w))).ravel()
            _, first = np.unique(v, return_index=True)
            packed = packed[np.sort(first)]

        b = np.unpackbits(packed[:n], axis=1, count=total)
        return (np.nonzero(b)[1].reshape(n, l) + low).astype(np.int32)

    def __getitem__(self, i):
        if not isinstance(i, (int, np.integer)):
            raise TypeError("Index must be an integer.")

        if i < 0:
            i += self.div

        if not 0 <= i < self.div:
            raise IndexError("Density index out of range.")

        if self.div == 1:
            trees = self.range
        else:
            trees = round(self.range * i / (self.div - 1))

        max_tag = math.comb(self.range, trees)
        target = min(self.num, max_tag) if self.max_tag else max_tag

        boards = self.generate_unique(
            n=target,
            l=trees,
            low=0,
            high=self.range
        )
        return self.convert(boards)

    def convert(self, i):
        boards = i
        matrices = np.zeros(
            (len(boards), self.bs * self.bs),
            dtype=np.uint8
        )
        rows = np.arange(len(boards))[:, None]
        matrices[rows, boards] = 1

        return matrices.reshape(-1, self.bs, self.bs)

    def __len__(self):
        return self.div
