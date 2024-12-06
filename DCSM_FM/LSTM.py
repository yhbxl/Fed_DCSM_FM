import torch.nn as nn
import torch

class ClientSelectorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ClientSelectorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the last LSTM output
        return torch.sigmoid(out)  # Output as probabilities

# Example usage
def select_clients_with_lstm(lstm_model, client_histories, threshold=0.5):
    # Input shape: (batch_size, time_steps, input_size)
    scores = lstm_model(client_histories)
    selected_clients = (scores > threshold).nonzero(as_tuple=True)[0]
    return selected_clients
