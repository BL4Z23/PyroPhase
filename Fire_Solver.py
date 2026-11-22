import numpy as np


class FireSolver:
    """
    Ignition: first tree in row-major order (topmost row, then leftmost).
    Spread:   4-neighbour.
    Target:   the fixed bottom row (row n-1).

    Returns 0/1 values as [spanning, complete]:
      spanning = fire reached the bottom row
      complete = every tree burned (independent of spanning)
    Empty board returns [0, 0].
    """

    def __init__(self, n=12):
        self.n = n
        cells = n * n
        col0 = sum(1 << (r * n) for r in range(n))
        colL = sum(1 << (r * n + n - 1) for r in range(n))
        self.not_col0 = ((1 << cells) - 1) & ~col0
        self.not_colL = ((1 << cells) - 1) & ~colL
        self.last_row = ((1 << n) - 1) << (n * (n - 1))

    def solve(self, board):
        n = self.n
        flat = np.asarray(board, dtype=np.uint8).reshape(-1)
        trees = int.from_bytes(
            np.packbits(flat, bitorder="little").tobytes(), "little"
        )
        if trees == 0:
            return np.zeros(2, dtype=np.uint8)

        fire = trees & -trees  # top-left-most tree
        while True:
            nxt = (
                          fire
                          | ((fire << 1) & self.not_col0)
                          | ((fire >> 1) & self.not_colL)
                          | (fire << n)
                          | (fire >> n)
                  ) & trees
            if nxt == fire:
                break
            fire = nxt

        return np.array(
            [bool(fire & self.last_row), fire == trees], dtype=np.uint8
        )

    def solve_batch(self, boards):
        n = self.n
        t = np.asarray(boards, dtype=bool).reshape(-1, n, n)
        N = len(t)
        flat = t.reshape(N, -1)

        has = flat.any(axis=1)
        ign = flat.argmax(axis=1)

        fire = np.zeros((N, n * n), dtype=bool)
        fire[np.arange(N), ign] = has
        fire = fire.reshape(N, n, n)

        while True:
            nxt = fire.copy()
            nxt[:, :, 1:] |= fire[:, :, :-1]
            nxt[:, :, :-1] |= fire[:, :, 1:]
            nxt[:, 1:, :] |= fire[:, :-1, :]
            nxt[:, :-1, :] |= fire[:, 1:, :]
            nxt &= t
            if np.array_equal(nxt, fire):
                break
            fire = nxt

        spanning = fire[:, -1, :].any(axis=1)
        complete = (fire == t).all(axis=(1, 2)) & has
        return np.stack([spanning, complete], axis=1).astype(np.uint8)
