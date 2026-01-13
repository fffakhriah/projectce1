import numpy as np

class ACO_Multi:
    def __init__(self, data, ants=20, iterations=50,
                 w1=0.6, w2=0.3, w3=0.1, evap=0.5, max_price=30):

        # Filter data: only prices <= RM30
        filtered_data = data[data['ticket_price'] <= max_price]

        self.price = filtered_data['ticket_price'].values
        self.demand = filtered_data['number_of_persons'].values

        self.pheromone = np.ones(len(self.price))

        self.ants = ants
        self.iterations = iterations
        self.w1 = w1      # revenue weight
        self.w2 = w2      # demand weight
        self.w3 = w3      # price penalty weight
        self.evap = evap
        self.max_price = max_price

    # Multi-objective fitness function
    def fitness(self, i):
        revenue = self.price[i] * self.demand[i]

        fitness_value = (
            self.w1 * revenue +
            self.w2 * self.demand[i] -
            self.w3 * self.price[i]   # price penalty
        )

        return fitness_value

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

            # Evaporation
            self.pheromone *= (1 - self.evap)

            # Pheromone update
            self.pheromone[best_idx] += best_score / 1000

            history.append(best_score)

        best_price = self.price[best_idx]
        best_customers = self.demand[best_idx]

        return best_price, best_customers, best_score, history
