from genetic import Algorithm
import random


class Test(Algorithm):

    def estimate(self, g):
        return (g[0] - 5)**2 + (g[1] + 3)**2

    def mutate(self, g):
        ind = random.choice(range(len(g)))
        g[ind] = random.uniform(-7, 7)
        return g


t = Test(mutate_prob=0.5)
start = [
    [random.uniform(-7, 7), random.uniform(-7, 7)]
    for _ in range(32)
]

result = t.run(50, start)

print(result)
