import heapq

class TimeWindowSelector:
    def __init__(self, window_size, num_clients):
        self.window_size = window_size
        self.num_clients = num_clients
        self.client_performances = [[] for _ in range(num_clients)]

    def update_client_performance(self, client_id, performance):
        if len(self.client_performances[client_id]) >= self.window_size:
            self.client_performances[client_id].pop(0)
        self.client_performances[client_id].append(performance)

    def select_clients(self, clients, num_clients):
        average_performances = [(i, np.mean(perfs) if perfs else 0) for i, perfs in enumerate(self.client_performances)]
        selected_clients_indices = [client_id for client_id, _ in heapq.nlargest(num_clients, average_performances, key=lambda x: x[1])]
        return selected_clients_indices