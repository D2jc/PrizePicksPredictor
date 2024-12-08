import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)  # Attention weights
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # LSTM output
        lstm_out, _ = self.lstm(x)  # lstm_out: [batch_size, seq_len, hidden_size]
        
        # Attention mechanism
        attn_weights = self.attention(lstm_out)  # [batch_size, seq_len, 1]
        attn_weights = F.softmax(attn_weights, dim=1)  # Normalize weights across the sequence
        
        # Weighted sum of LSTM outputs
        context_vector = torch.sum(lstm_out * attn_weights, dim=1)  # [batch_size, hidden_size]
        
        # Dropout and fully connected layer
        output = self.fc(context_vector)
        return self.sigmoid(output)
