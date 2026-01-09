import numpy as np

class ACO_Single:
    def __init__(self, data, ants=20, iterations=50, evap=0.5):
        self.price = data['ticket_price'].values
        self.demand = data['number_of_persons'].values
        self.pheromone = np.ones(len(self.price))

        self.ants = ants
        self.iterations = iterations
        self.evap = evap

    def fitness(self, i):
        # SINGLE OBJECTIVE: MAXIMIZE REVENUE
        return self.price[i] * self.demand[i]

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
