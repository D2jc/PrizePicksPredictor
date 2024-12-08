import torch
import torch.nn as nn

class StackedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.5, bidirectional=False):
        super(StackedLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        # Stacked LSTM layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # LSTM output
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Get the last hidden state of the last layer
        # h_n: [num_layers * num_directions, batch_size, hidden_size]
        last_hidden_state = h_n[-self.num_directions:, :, :]  # [num_directions, batch_size, hidden_size]
        last_hidden_state = last_hidden_state.transpose(0, 1).reshape(x.size(0), -1)  # [batch_size, hidden_size * num_directions]

        # Fully connected layer
        output = self.fc(last_hidden_state)
        return self.sigmoid(output)
