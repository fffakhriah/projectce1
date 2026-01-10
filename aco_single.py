import numpy as np

class ACO_Single:
    def __init__(self, data, ants=20, iterations=50, evap=0.5):
        self.price = data['ticket_price'].values
        self.demand = data['number_of_persons'].values
        self.pheromone = np.ones(len(self.price))

        self.ants = persons
        self.iterations = iterations
        self.evap = evap

    def run(self):
        best_idx = 0
        best_fitness = 0
        history = []

        for _ in range(self.iterations):
            probs = self.pheromone / self.pheromone.sum()
            choices = np.random.choice(len(self.price), self.persons, p=probs)

            for idx in choices:
                revenue = self.price[idx] * self.demand[idx]
                if revenue > best_fitness:
                    best_fitness = revenue
                    best_idx = idx

            self.pheromone *= (1 - self.evap)
            self.pheromone[best_idx] += best_fitness / 1000
            history.append(best_fitness)

        return self.price[best_idx], best_fitness, history
