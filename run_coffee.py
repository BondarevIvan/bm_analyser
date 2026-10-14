from magnum.magnum import (
    load_model_file,
    run_model
)
import matplotlib.pyplot as plt


model = load_model_file("models/coffee.mgn")

results = []

for seed in range(100000, 100000+200):
    res = run_model(model, horizon=30, seed=seed)
    count_processed, count_lost = res['variables']['count_processed']['value'][-1], res['variables']['count_lost']['value'][-1]
    print(count_processed, count_lost)
    results.append(count_processed)


plt.hist(results, bins=10)
plt.savefig("a.png")
