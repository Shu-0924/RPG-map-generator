import numpy as np


def compute_fitness(population):
    fitness_value = []
    for m in population:
        n = m.shape[0]
        obstacle_state = np.array([[min(2, (
            int(i == 0 or m[i-1][j] != 0) + int(i == n-1 or m[i+1][j] != 0)
            + int(j == 0 or m[i][j-1] != 0) + int(j == n-1 or m[i][j+1] != 0)
            if m[i][j] == 0 else 0)) for j in range(n)] for i in range(n)], dtype=np.int32)
        fitness_value.append(np.sum(obstacle_state))
    return np.asarray(fitness_value, dtype=np.int32)


def tournament_selection(population, tournament_size=2):
    fitness_value = np.asarray(compute_fitness(population), dtype=np.int32)

    select_index = []
    for _ in range(population.shape[0]):
        idx = np.random.choice(population.shape[0], tournament_size, replace=False)
        select_index.append(idx[0] if fitness_value[idx[0]] >= fitness_value[idx[1]] else idx[1])

    select_index = np.asarray(select_index)
    return population[select_index]


def recombination(parents):
    crossover_points = np.random.randint(1, parents.shape[1], size=parents.shape[0]//2)
    parents_1 = parents[np.arange(0, parents.shape[0], 2)]
    parents_2 = parents[np.arange(1, parents.shape[0], 2)]

    children = []
    for i in range(crossover_points.shape[0]):
        idx = crossover_points[i]
        children.append(np.concatenate((parents_1[i][:idx, :], parents_2[i][idx:, :])))
        children.append(np.concatenate((parents_2[i][:idx, :], parents_1[i][idx:, :])))

    mutation_points = np.random.randint(1, parents.shape[1], size=parents.shape[0] // 2)
    for i in range(mutation_points.shape[0]):
        mutation_child = children[i].copy()
        tmp = mutation_child[mutation_points[i]]
        rnd_result = np.random.random()
        if rnd_result < 0.05:
            tmp = np.zeros_like(tmp)
        elif rnd_result < 0.1:
            tmp = np.ones_like(tmp)
        elif rnd_result < 0.15:
            tmp = 1 - tmp
        else:
            idx = np.random.choice(tmp.shape[0], 2, replace=False)
            tmp[idx[0]], tmp[idx[1]] = tmp[idx[1]], tmp[idx[0]]
        mutation_child[mutation_points[i]] = tmp
        children.append(mutation_child)

    children_fitness = compute_fitness(children)
    sorted_idx = np.argsort(children_fitness, )[-parents.shape[0]:]
    return np.asarray(children)[sorted_idx]


def run(n, population_size=250, generation=100):
    population = []
    for _ in range(population_size):
        population.append(np.random.randint(0, 2, size=(n, n)))
    population = np.asarray(population, dtype=np.int32)

    for i in range(generation):
        parents = tournament_selection(population)
        children = recombination(parents)
        population = children
        print("Iteration{} - Current Best Fitness: {}".format(i+1, compute_fitness(population).max()))

    return population


def save_map(population, n=10):
    fitness_value = compute_fitness(population)
    idx = np.random.choice(population.shape[0], n, replace=False)
    for i in range(n):
        print("Randomly choose the map with {} fitness value".format(fitness_value[idx[i]]))
        m = [''.join([('2' if x == 0 else '3') for x in row]) for row in population[idx[i]]]
        with open("./RPGGame/data/generate{}.map".format(i+1), "w") as f:
            for row in m:
                f.write(row + '\n')


if __name__ == '__main__':
    result = run(n=46)
    save_map(result)
