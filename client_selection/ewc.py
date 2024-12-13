import random

class EWCClientSelector:
    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.client_scores = {}

    def update_client_score(self, client_id, score):
        self.client_scores[client_id] = score

    def select_clients(self, clients):
        if not self.client_scores:
            # 如果没有分数，随机选择
            selected_clients_indices = random.sample(range(len(clients)), self.num_clients)
        else:
            # 根据分数选择
            sorted_clients = sorted(self.client_scores.items(), key=lambda item: item[1], reverse=True)
            selected_clients_indices = [client_id for client_id, _ in sorted_clients[:self.num_clients]]
        return selected_clients_indices