import random

class TimeWindowSelector:
    def __init__(self, window_size):
        self.window_size = window_size
        self.client_histories = {}

    def update_client_performance(self, client_id, performance_score):
        if client_id not in self.client_histories:
            self.client_histories[client_id] = []
        history = self.client_histories[client_id]
        history.append(performance_score)
        if len(history) > self.window_size:
            history.pop(0)

    def select_clients(self, clients, num_clients):
        # 示例：随机选择客户端
        selected_clients_indices = random.sample(range(len(clients)), num_clients)
        return selected_clients_indices