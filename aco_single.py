import numpy as np

class ACO_Single:
    def __init__(self, data, ants=20, iterations=50, evap=0.5, max_revenue=250):
        self.price = data['ticket_price'].values
        self.demand = data['number_of_persons'].values
        self.pheromone = np.ones(len(self.price))

        self.ants = ants
        self.iterations = iterations
        self.evap = evap
        self.max_revenue = max_revenue   # revenue limit

    def run(self):
        best_idx = 0
        best_fitness = -1e9
        history = []

        for _ in range(self.iterations):
            probs = self.pheromone / self.pheromone.sum()
            choices = np.random.choice(len(self.price), self.ants, p=probs)

            for idx in choices:
                revenue = self.price[idx] * self.demand[idx]

                # Apply revenue limit
                if revenue > self.max_revenue:
                    revenue = self.max_revenue

                if revenue > best_fitness:
                    best_fitness = revenue
                    best_idx = idx

            # Evaporation
            self.pheromone *= (1 - self.evap)

            # Update pheromone on best solution
            self.pheromone[best_idx] += best_fitness / 1000

            history.append(best_fitness)

        best_price = self.price[best_idx]
        best_customers = self.demand[best_idx]

        return best_price, best_customers, best_fitness, history
