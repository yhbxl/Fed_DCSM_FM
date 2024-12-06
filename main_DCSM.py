# 导入所需模块
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from DCSM_FM.EWC import EWCClient
from DCSM_FM.LSTM import ClientSelectorLSTM
from DCSM_FM.TimeWindow import TimeWindowSelector


# 联邦学习训练过程函数
def fedavg_train(server_model, clients, num_rounds=10, num_clients=5, epochs=1, lr=0.01):
    # 初始化时间窗口选择器
    time_window_selector = TimeWindowSelector(window_size=10)
    # 初始化 LSTM 模型用于客户端选择
    lstm_model = ClientSelectorLSTM(input_size=5, hidden_size=10, output_size=1)

    # 将服务器模型和 LSTM 模型移动到 GPU
    server_model.to(device)
    lstm_model.to(device)

    for round_num in range(num_rounds):
        print(f"Round {round_num + 1}")

        # Step 1: 使用 LSTM 模型选择客户端
        client_histories = torch.rand(len(clients), 10, 5)  # 示例输入：随机生成的客户端历史记录
        selected_clients_indices = select_clients_with_lstm(lstm_model, client_histories)[:num_clients]

        # Step 2: 在选定客户端上训练本地模型
        client_weights = []
        accuracies = []
        losses = []
        for client_id in selected_clients_indices:
            client = clients[client_id]
            client_weight, accuracy, loss = client.train(server_model, epochs=epochs, lr=lr)
            client_weights.append((client_weight, len(client.data_loader.dataset)))
            accuracies.append(accuracy)
            losses.append(loss)

        print("Selected clients:", selected_clients_indices)
        print(f"Average Accuracy: {np.mean(accuracies)}")
        print(f"Average Loss: {np.mean(losses)}")

        # Step 3: 聚合权重（FedAvg）
        total_samples = sum([w[1] for w in client_weights])
        new_global_weights = {key: torch.zeros_like(val)
                              for key, val in server_model.state_dict().items()}
        for client_weight, num_samples in client_weights:
            for key in new_global_weights:
                new_global_weights[key] += client_weight[key] * (num_samples / total_samples)

        # 更新全局模型权重
        server_model.load_state_dict(new_global_weights)

        # Step 4: 更新时间窗口
        for client_id in selected_clients_indices:
            # 示例：性能分数可以定义为使用的数据大小
            time_window_selector.update_client_performance(client_id, len(clients[client_id].data_loader.dataset))

    print("Training completed!")


# 示例客户端训练类
class Client:
    def __init__(self, data_loader, device):
        self.data_loader = data_loader
        self.device = device

    def train(self, model, epochs, lr):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        total_loss = 0
        correct = 0
        total = 0

        for epoch in range(epochs):
            for data, target in self.data_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        # 计算平均损失和准确率
        avg_loss = total_loss / total
        accuracy = correct / total

        # 返回客户端的模型权重、准确率和损失
        return copy.deepcopy(model.state_dict()), accuracy, avg_loss


# 使用 LSTM 模型选择客户端的函数
def select_clients_with_lstm(lstm_model, client_histories):
    lstm_model.eval()
    scores = lstm_model(client_histories)  # 假设输出是客户端得分
    selected_clients = scores.squeeze().topk(k=len(client_histories), dim=0).indices.tolist()  # 选出得分最高的客户端
    return selected_clients


# 主函数
if __name__ == "__main__":
    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 示例：加载 MNIST 数据集
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('../data/mnist/', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data/mnist/', train=False, download=True, transform=transform)

    # 将数据集划分给多个客户端
    num_clients = 10
    data_loaders = [torch.utils.data.DataLoader(torch.utils.data.Subset(train_dataset, range(i, len(train_dataset), num_clients)), batch_size=32, shuffle=True) for i in range(num_clients)]

    # 初始化全局模型（MLP 示例）
    global_model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(28 * 28, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10)
    )

    # 将客户端封装成对象
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clients = [Client(data_loader, device) for data_loader in data_loaders]

    # 启动联邦训练
    fedavg_train(global_model, clients, num_rounds=5, num_clients=5, epochs=1, lr=0.01)

    print("Global DCSM_FM training completed!")



