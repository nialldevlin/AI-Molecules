import torch
import torch.nn as nn


class CNNtoRNN(nn.Module):
    # TODO make this use the actual data size
    def __init__(self, channels=3, cnn_output_size=256, cnn_hidden_size=128,
                 lstm_hidden_size=256, num_lstm_layers=2, num_cnn_layers=2):
        super(CNNtoRNN, self).__init__()

        # Initialize CNN layers dynamically based on num_cnn_layers
        cnn_layers = []
        for i in range(num_cnn_layers):
            in_channels = channels if i == 0 else cnn_hidden_size
            cnn_layers.extend([
                nn.Conv1d(in_channels=in_channels,
                          out_channels=cnn_hidden_size,
                          kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            ])

        cnn_layers.extend([
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(cnn_hidden_size, cnn_output_size),
            nn.ReLU()
        ])

        self.cnn = nn.Sequential(*cnn_layers)

        # LSTM component for processing temporal sequences of CNN outputs
        self.lstm = nn.LSTM(input_size=cnn_output_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=num_lstm_layers,
                            batch_first=True)

        # Fully connected layer to predict the next timestep
        self.fc = nn.Linear(lstm_hidden_size, 250*3)

    def forward(self, x):
        # Adjusted for input shape [batch_size, timesteps, length, channels]
        batch_size, timesteps, L, C = x.size()
        # Process each timestep through the CNN
        c_out = []
        for t in range(timesteps):
            # Extract the timestep across all batches
            # Shape: [batch_size, length, channels]
            timestep_input = x[:, t, :, :]
            # Reshape to [batch_size, channels, length] for Conv1d
            timestep_input = timestep_input.permute(0, 2, 1)
            c_out_t = self.cnn(timestep_input)
            c_out.append(c_out_t)

        # Stack the CNN outputs for each timestep
        # Shape: [batch_size, timesteps, cnn_output_size]
        c_out = torch.stack(c_out, dim=1)

        # Pass the sequence of CNN outputs through the LSTM
        r_out, (h_n, c_n) = self.lstm(c_out)

        # Use the last hidden state to predict the next timestep
        predictions = self.fc(r_out[:, -1, :])
        return predictions.view(batch_size, 250, 3)
