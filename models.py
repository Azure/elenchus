import torch
from types import SimpleNamespace

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim, dtype=torch.float64),
            torch.nn.Linear(hidden_dim, output_dim, dtype=torch.float64)
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def forward(self, embedding=None, labels=None):
        logits = self.layers(embedding)
        loss = self.criterion(logits, labels)

        outputs = SimpleNamespace(logits=logits, loss=loss)

        return outputs
