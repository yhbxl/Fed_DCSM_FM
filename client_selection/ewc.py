import random

class EWCClientSelector:
    def __init__(self, num_clients):
        """
        初始化 EWC 客户端选择器
        :param num_clients: 需要选择的客户端数量
        """
        self.num_clients = num_clients
        self.client_scores = {}  # 存储客户端的分数，格式为 {client_id: score}

    def update_client_score(self, client_id, score, weight=1.0):
        """
        更新客户端的分数，支持多指标加权更新
        :param client_id: 客户端的唯一标识
        :param score: 该客户端的得分
        :param weight: 更新权重，默认为 1.0
        """
        self.client_scores[client_id] = self.client_scores.get(client_id, 0) + weight * score

    def decay_scores(self, decay_factor=0.9):
        """
        对所有客户端的分数进行衰减，模拟时间变化的影响
        :param decay_factor: 衰减系数，范围 (0, 1)，越小衰减越快
        """
        for client_id in self.client_scores:
            self.client_scores[client_id] *= decay_factor

    def select_clients(self, clients, explore_prob=0.1):
        """
        根据分数选择客户端，加入一定的随机探索机制
        :param clients: 客户端列表，包含所有可用客户端的 ID
        :param explore_prob: 随机探索概率，范围 [0, 1]，越大越倾向于随机选择
        :return: 被选中的客户端列表
        """
        if random.random() < explore_prob or not self.client_scores:
            # 随机探索
            selected_clients_indices = random.sample(range(len(clients)), self.num_clients)
            selected_clients = [clients[i] for i in selected_clients_indices]
        else:
            # 利用分数选择
            sorted_clients = sorted(self.client_scores.items(), key=lambda item: item[1], reverse=True)
            selected_clients = [client_id for client_id, _ in sorted_clients[:self.num_clients]]
        return selected_clients

if __name__ == '__main__':
    # 初始化选择器，选择 x 个客户端
    selector = EWCClientSelector(num_clients=5)

    # 假设有 5 个客户端，客户端 ID 为 0-4
    clients = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # 更新客户端的分数
    selector.update_client_score(0, 0.85, weight=1.0)
    selector.update_client_score(1, 0.75, weight=0.8)
    selector.update_client_score(2, 0.90, weight=1.2)
    selector.update_client_score(3, 0.60, weight=0.9)
    selector.update_client_score(4, 0.80, weight=1.0)

    # 对分数进行衰减
    selector.decay_scores(decay_factor=0.95)

    # 根据分数和随机探索机制选择客户端
    selected_clients = selector.select_clients(clients, explore_prob=0.2)
    print("Selected Clients:", selected_clients)

    # 查看当前客户端分数
    print("Client Scores:", selector.client_scores)
