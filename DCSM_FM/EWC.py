import torch
import torch.nn as nn
import torch.optim as optim

class EWCClient:
    def __init__(self, model, data_loader, ewc_lambda, fisher_information, optimal_params):
        self.model = model
        self.data_loader = data_loader
        self.ewc_lambda = ewc_lambda
        self.fisher_information = fisher_information
        self.optimal_params = optimal_params

    def train(self, global_model, epochs=1, lr=0.01):
        # Load global DCSM_FM
        self.model.load_state_dict(global_model.state_dict())
        optimizer = optim.SGD(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for data, target in self.data_loader:
                optimizer.zero_grad()

                # Compute task loss
                output = self.model(data)
                task_loss = criterion(output, target)

                # Compute EWC regularization
                ewc_loss = 0
                for param, fisher, optimal_param in zip(
                        self.model.parameters(),
                        self.fisher_information,
                        self.optimal_params):
                    ewc_loss += (fisher * (param - optimal_param) ** 2).sum()

                # Total loss
                total_loss = task_loss + self.ewc_lambda * ewc_loss
                total_loss.backward()
                optimizer.step()

        return self.model.state_dict()
