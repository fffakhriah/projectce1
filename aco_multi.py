import numpy as np

class ACO_Multi:
    def __init__(
        self, data, ants=20, iterations=50,
        w1=0.6, w2=0.3, w3=0.1, evap=0.5
    ):
        self.price = data['ticket_price'].values
        self.demand = data['number_of_persons'].values
        self.pheromone = np.ones(len(self.price))

        self.ants = ants
        self.iterations = iterations
        self.evap = evap

        # Weights for multi-objective fitness
        self.w1 = w1  # revenue importance
        self.w2 = w2  # demand importance
        self.w3 = w3  # price penalty

    def fitness(self, i):
        revenue = self.price[i] * self.demand[i]

        # MULTI-OBJECTIVE FITNESS (BALANCED)
        return (
            self.w1 * revenue +
            self.w2 * self.demand[i] -
            self.w3 * self.price[i]
        )

    def run(self):
        best_price, best_fitness = None, -np.inf
        history = []

        for _ in range(self.iterations):
            selected, fitness_vals = [], []

            prob = self.pheromone / self.pheromone.sum()

            for _ in range(self.ants):
                i = np.random.choice(len(self.price), p=prob)
                f = self.fitness(i)

                selected.append(i)
                fitness_vals.append(f)

                if f > best_fitness:
                    best_fitness = f
                    best_price = self.price[i]

            self.pheromone *= (1 - self.evap)

            for i, f in zip(selected, fitness_vals):
                self.pheromone[i] += f / 1000

            history.append(best_fitness)

        return best_price, best_fitness, history
