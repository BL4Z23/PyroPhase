import csv
import math
import os
import secrets
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from Board_Generator import BoardGenerator
from Fire_Solver import FireSolver

SIZE_OF_BOARDS = 100
NUMBER_OF_SAMPLES = 10000
BATCH_SIZE = 500
DIVISIONS = 1000
CAPPED_GENERATION = True
SEED = 42
WORKERS = os.cpu_count() or 1
HEADERS = ['PERCENTAGE COVERAGE', 'SPANNED PERCENTAGE', 'COMPLETE PERCENTAGE']

_solver = None
_generator = None
_seed = None


def init_worker(seed):
    global _solver, _generator, _seed
    _seed = seed
    _solver = FireSolver(n=SIZE_OF_BOARDS)
    _generator = BoardGenerator(
        board_size=SIZE_OF_BOARDS,
        number_of_boards=BATCH_SIZE,
        divisions=DIVISIONS,
        max_tag=CAPPED_GENERATION,
        seed=seed,
    )


def run_batch(task):
    assert _generator is not None and _solver is not None
    division, batch = task
    _generator.rng = np.random.default_rng([_seed, division, batch])
    boards = _generator[division]
    solution = _solver.solve_batch(boards)
    trees = int(boards[0].sum())
    return (
        division,
        trees,
        len(boards),
        int(solution[:, 0].sum()),
        int(solution[:, 1].sum()),
    )


def trees_for(division):
    cells = SIZE_OF_BOARDS ** 2
    if DIVISIONS == 1:
        return cells
    return round(cells * division / (DIVISIONS - 1))


def build_tasks():
    cells = SIZE_OF_BOARDS ** 2
    tasks = []
    for division in range(DIVISIONS):
        possible = math.comb(cells, trees_for(division))
        batches = 1 if possible <= BATCH_SIZE else math.ceil(NUMBER_OF_SAMPLES / BATCH_SIZE)
        for batch in range(batches):
            tasks.append((division, batch))
    return tasks


def main():
    seed = SEED if SEED is not None else secrets.randbits(64)
    tasks = build_tasks()
    totals = defaultdict(lambda: [0, 0, 0, 0])

    with ProcessPoolExecutor(
            max_workers=WORKERS,
            initializer=init_worker,
            initargs=(seed,),
    ) as pool:
        futures = [pool.submit(run_batch, t) for t in tasks]
        for i, f in enumerate(as_completed(futures), 1):
            division, trees, count, spanned, complete = f.result()
            t = totals[division]
            t[0] = trees
            t[1] += count
            t[2] += spanned
            t[3] += complete
            if i % 100 == 0 or i == len(futures):
                print(f"{i}/{len(futures)} batches", flush=True)

    cells = SIZE_OF_BOARDS ** 2
    with open("output.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)
        for division in sorted(totals):
            trees, count, spanned, complete = totals[division]
            writer.writerow([
                100 * trees / cells,
                100 * spanned / count,
                100 * complete / count,
            ])


if __name__ == "__main__":
    main()
