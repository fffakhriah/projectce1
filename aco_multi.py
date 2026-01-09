import numpy as np

class ACO_Multi:
    def __init__(self, data, ants=20, iterations=50,
                 w1=0.6, w2=0.3, w3=0.1, evap=0.5):

        self.price = data['ticket_price'].values
        self.demand = data['number_of_persons'].values
        self.pheromone = np.ones(len(self.price))

        self.ants = ants
        self.iterations = iterations
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.evap = evap

    def fitness(self, i):
        revenue = self.price[i] * self.demand[i]
        return (self.w1 * revenue +
                self.w2 * self.demand[i] -
                self.w3 * self.price[i])

    def run(self):
        best_idx = 0
        best_score = -1e9
        history = []

        for _ in range(self.iterations):
            probs = self.pheromone / self.pheromone.sum()
            choices = np.random.choice(len(self.price), self.ants, p=probs)

            for idx in choices:
                score = self.fitness(idx)
                if score > best_score:
                    best_score = score
                    best_idx = idx

            self.pheromone *= (1 - self.evap)
            self.pheromone[best_idx] += abs(best_score) / 1000
            history.append(best_score)

        return self.price[best_idx], best_score, history
