# 导入所需模块
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import yaml
import os

# 导入客户端选择策略
from client_selection.EWC import EWCClientSelector
from client_selection.LSTM import ClientSelectorLSTM, select_clients_with_lstm
from client_selection.TimeWindow import TimeWindowSelector

# 加载配置文件
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

strategy = config['strategy']
num_clients = config['num_clients']
num_rounds = config['num_rounds']
epochs = config['epochs']
lr = config['lr']

if strategy == 'lstm':
    lstm_config = config['lstm_config']
    lstm_model = ClientSelectorLSTM(**lstm_config)
elif strategy == 'timewindow':
    time_window_config = config['time_window_config']
    time_window_selector = TimeWindowSelector(**time_window_config)
else:
    ewc_client_selector = EWCClientSelector(num_clients=num_clients)

# 联邦学习训练过程函数
def fedavg_train(server_model, clients, num_rounds=num_rounds, num_clients=num_clients, epochs=epochs, lr=lr):
    if strategy == 'lstm':
        lstm_model.to(device)
    elif strategy == 'timewindow':
        pass  # 时间窗口选择器不需要移动到设备

    for round_num in range(num_rounds):
        print(f"Round {round_num + 1}")

        # Step 1: 使用指定的客户端选择策略选择客户端
        if strategy == 'lstm':
            client_histories = torch.rand(len(clients), lstm_config['window_size'], lstm_config['input_size']).to(device)  # 示例输入：随机生成的客户端历史记录
            selected_clients_indices = select_clients_with_lstm(lstm_model, client_histories)[:num_clients]
        elif strategy == 'timewindow':
            selected_clients_indices = time_window_selector.select_clients(clients, num_clients)
        else:
            selected_clients_indices = ewc_client_selector.select_clients(clients)

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

        # Step 4: 计算全局平均损失
        global_loss = 0
        global_correct = 0
        global_total = 0
        for client_id in selected_clients_indices:
            client = clients[client_id]
            client_model = copy.deepcopy(server_model)
            client_model.eval()
            loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

            with torch.no_grad():
                for data, target in client.data_loader:
                    data, target = data.to(device), target.to(device)
                    output = client_model(data)
                    loss = loss_fn(output, target)
                    global_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    global_total += target.size(0)
                    global_correct += (predicted == target).sum().item()

        global_avg_loss = global_loss / global_total
        global_accuracy = global_correct / global_total
        print(f"Global Average Loss: {global_avg_loss}")
        print(f"Global Accuracy: {global_accuracy}")

        # Step 5: 更新时间窗口
        if strategy == 'timewindow':
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
        model.to(self.device)  # 将模型移动到 GPU
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        total_loss = 0
        correct = 0
        total = 0

        for epoch in range(epochs):
            for data, target in self.data_loader:
                data, target = data.to(self.device), target.to(self.device)  # 将数据移动到 GPU
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

# 主函数
if __name__ == "__main__":
    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 示例：加载 MNIST 数据集
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data/mnist/', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data/mnist/', train=False, download=True, transform=transform)

    # 将数据集划分给多个客户端
    num_clients = config['num_clients']
    data_loaders = [torch.utils.data.DataLoader(torch.utils.data.Subset(train_dataset, range(i, len(train_dataset), num_clients)), batch_size=32, shuffle=True) for i in range(num_clients)]

    # 初始化全局模型（MLP 示例）
    global_model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(28 * 28, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10)
    )

    # 将客户端封装成对象
    clients = [Client(data_loader, device) for data_loader in data_loaders]

    # 启动联邦训练
    fedavg_train(global_model, clients, num_rounds=num_rounds, num_clients=num_clients, epochs=epochs, lr=lr)

    print("Global model training completed!")