class TimeWindowSelector:
    def __init__(self, window_size):
        self.window_size = window_size
        self.history = {}

    def update_client_performance(self, client_id, performance_score):
        if client_id not in self.history:
            self.history[client_id] = []
        self.history[client_id].append(performance_score)

        # Trim history to window size
        if len(self.history[client_id]) > self.window_size:
            self.history[client_id] = self.history[client_id][-self.window_size:]

    def select_clients(self, num_clients):
        # Compute average performance over the window
        avg_scores = {client_id: sum(scores) / len(scores)
                      for client_id, scores in self.history.items()}
        # Sort clients by performance
        sorted_clients = sorted(avg_scores, key=avg_scores.get, reverse=True)
        return sorted_clients[:num_clients]
