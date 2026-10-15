from concurrent.futures import ProcessPoolExecutor

from magnum.magnum import (
    load_model_file,
    run_model
)
import matplotlib.pyplot as plt

MODEL_PATH = "models/coffee.mgn"
SEEDS = list(range(100000, 100000 + 3000))
HORIZON = 30
THREADS = 8  # число параллельных процессов; сиды делятся между ними поровну


def run_seeds(seeds):
    """Прогнать модель на части сидов (выполняется в отдельном процессе)."""
    model = load_model_file(MODEL_PATH)  # каждый процесс загружает модель сам
    out = []
    for seed in seeds:
        res = run_model(model, horizon=HORIZON, seed=seed, debug=False,
                        record=["count_processed", "count_lost"])
        count_processed = res['variables']['count_processed']['value'][-1]
        count_lost = res['variables']['count_lost']['value'][-1]
        out.append((seed, count_processed, count_lost))
    return out


def split(items, n):
    """Разбить список на n почти равных частей."""
    k, m = divmod(len(items), n)
    return [items[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


if __name__ == "__main__":  # обязательно для multiprocessing на macOS/Windows
    with ProcessPoolExecutor(max_workers=THREADS) as pool:
        chunks = pool.map(run_seeds, split(SEEDS, THREADS))
        rows = [row for chunk in chunks for row in chunk]

    processed = []
    lost = []
    for seed, count_processed, count_lost in rows:
        processed.append(count_processed)
        lost.append(count_lost)

    plt.hist(processed, bins=20)
    plt.savefig("processed.png")
    plt.clf()

    plt.hist(lost, bins=20)
    plt.savefig("lost.png")
    plt.clf()
