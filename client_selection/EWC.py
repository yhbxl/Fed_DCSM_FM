import torch.nn as nn
import torch.optim as optim
import random

class EWCClientSelector:
    def __init__(self, num_clients):
        self.num_clients = num_clients

    def select_clients(self, clients):
        # 示例：随机选择客户端
        selected_clients_indices = random.sample(range(len(clients)), self.num_clients)
        return selected_clients_indices