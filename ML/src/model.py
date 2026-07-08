from torch import nn

class BodyFatMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes = [64, 32], output_size=1, dropout_rate=0.2):
        super(BodyFatMLP, self).__init__()
        
        layers = []
        previous_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(previous_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            previous_size = hidden_size

        layers.append(nn.Linear(previous_size, output_size))

        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    