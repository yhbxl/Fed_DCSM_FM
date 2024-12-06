import torch
import torch.nn as nn
import random


class ClientSelectorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ClientSelectorLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out


def select_clients_with_lstm(lstm_model, client_histories, num_clients):
    lstm_model.eval()
    scores = lstm_model(client_histories)  # 假设输出是客户端得分
    selected_clients = scores.squeeze().topk(k=num_clients, dim=0).indices.tolist()  # 选出得分最高的客户端
    return selected_clients