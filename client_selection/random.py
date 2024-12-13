import random

class RandomClientSelector:
    def __init__(self, num_clients):
        self.num_clients = num_clients

    def select_clients(self, clients, num_clients):
        selected_clients_indices = random.sample(range(len(clients)), num_clients)
        return selected_clients_indices