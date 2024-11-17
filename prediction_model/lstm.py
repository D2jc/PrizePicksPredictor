import torch
import torch.nn as nn
import torch.optim as optim

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        LSTM model for sequential data.
        :param input_size: Number of features in the input
        :param hidden_size: Number of hidden units in the LSTM
        :param output_size: Number of output units
        :param learning_rate: Learning rate for the optimizer
        """
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # forget gate
        self.wlr1 = nn.Parameter(torch.randn(hidden_size, hidden_size))  # h_t * W
        self.wlr2 = nn.Parameter(torch.randn(input_size, hidden_size))  # x_t * W
        self.blr1 = nn.Parameter(torch.zeros(hidden_size))  # Bias

        # input gate
        self.wpr1 = nn.Parameter(torch.randn(hidden_size, hidden_size))  # h_t * W
        self.wpr2 = nn.Parameter(torch.randn(input_size, hidden_size))  # x_t * W
        self.bpr1 = nn.Parameter(torch.zeros(hidden_size))  # Bias

        # cell state update
        self.wp1 = nn.Parameter(torch.randn(hidden_size, hidden_size))  # h_t * W
        self.wp2 = nn.Parameter(torch.randn(input_size, hidden_size))  # x_t * W
        self.bp1 = nn.Parameter(torch.zeros(hidden_size))  # Bias

        # output gate
        self.wo1 = nn.Parameter(torch.randn(hidden_size, hidden_size))  # h_t * W
        self.wo2 = nn.Parameter(torch.randn(input_size, hidden_size))  # x_t * W
        self.bo1 = nn.Parameter(torch.zeros(hidden_size))  # Bias

        # fully connected layer for final output
        self.fc = nn.Linear(hidden_size, output_size)

    def lstm_unit(self, input_value, long_memory, short_memory):
        """
        A single LSTM unit computation for one timestep.
        :param input_value: Input at the current timestep (batch_size, input_size)
        :param long_memory: Previous long-term memory (batch_size, hidden_size)
        :param short_memory: Previous short-term memory (batch_size, hidden_size)
        :return: Updated long-term memory and short-term memory
        """
        # forget gate
        long_remember_percent = torch.sigmoid(
            torch.matmul(short_memory, self.wlr1) + torch.matmul(input_value, self.wlr2) + self.blr1
        )

        # input gate
        potential_remember_percent = torch.sigmoid(
            torch.matmul(short_memory, self.wpr1) + torch.matmul(input_value, self.wpr2) + self.bpr1
        )
        potential_memory = torch.tanh(
            torch.matmul(short_memory, self.wp1) + torch.matmul(input_value, self.wp2) + self.bp1
        )

        # update cell state
        updated_long_memory = (long_memory * long_remember_percent) + (potential_remember_percent * potential_memory)

        # output gate
        output_percent = torch.sigmoid(
            torch.matmul(short_memory, self.wo1) + torch.matmul(input_value, self.wo2) + self.bo1
        )
        updated_short_memory = output_percent * torch.tanh(updated_long_memory)

        return updated_long_memory, updated_short_memory

    def forward(self, input):
        """
        Forward pass through the LSTM.
        :param input: Input sequence (batch_size, seq_len, input_size)
        :return: Output of the LSTM (batch_size, output_size)
        """
        batch_size = input.size(0)
        seq_len = input.size(1)

        # Initialize memory states
        long_memory = torch.zeros(batch_size, self.hidden_size).to(input.device)
        short_memory = torch.zeros(batch_size, self.hidden_size).to(input.device)

        # Process each timestep sequentially
        for t in range(seq_len):
            x_t = input[:, t, :]  # Extract input at timestep t
            long_memory, short_memory = self.lstm_unit(x_t, long_memory, short_memory)

        # Use the final hidden state for the output
        final_output = self.fc(short_memory)
        return final_output

    def configure_optimizers(self):
        """
        Configure the optimizer for training.
        :return: Optimizer
        """
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer