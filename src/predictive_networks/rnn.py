import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedSpatialTemporalRNN(nn.Module):
    def __init__(self, input_channels=250, conv_out_channels=64, rnn_hidden_dim=128, sequence_length=10):
        super(EnhancedSpatialTemporalRNN, self).__init__()
        
        self.sequence_length = sequence_length
        # Convolutional layer for spatial feature extraction
        self.conv = nn.Conv2d(input_channels, conv_out_channels, kernel_size=3, padding=1)
        self.conv_output_size = conv_out_channels * 3 * 3  # Adjust based on actual output size
        
        # RNN layer to process temporal sequence
        self.rnn = nn.GRU(self.conv_output_size, rnn_hidden_dim, batch_first=True)
        
        # Linear layer to map RNN output back to spatial dimensions, then to original channel size
        self.to_original_size = nn.Linear(rnn_hidden_dim, input_channels * 3 * 3)
        
    def forward(self, x):
        # x is expected to be of shape [sequence_length, input_channels, H, W]
        # Process each frame through the convolutional layer
        conv_outputs = [F.relu(self.conv(x[t])) for t in range(self.sequence_length)]
        conv_outputs = torch.stack(conv_outputs, dim=0)  # Stack to form a batch
        conv_outputs = conv_outputs.view(self.sequence_length, -1)  # Flatten spatial dimensions
        
        # Reshape for RNN: (1, sequence_length, self.conv_output_size)
        # Since batch size is not considered, we treat the sequence as a batch of size 1
        rnn_input = conv_outputs.unsqueeze(0)
        
        # Process sequence with RNN
        rnn_out, _ = self.rnn(rnn_input)
        
        # Map RNN output back to spatial dimensions
        x = F.relu(self.to_original_size(rnn_out))
        x = x.view(self.sequence_length, input_channels, 3, 3)
        
        return x
