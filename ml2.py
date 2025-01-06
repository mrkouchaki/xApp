import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from influxdb import InfluxDBClient
import numpy as np

# Define parameters
seq_length = 10  # Number of time steps in the sequence
hidden_dim = 64  # Hidden dimension for the RNN
latent_dim = 32  # Dimension of the latent space
batch_size = 32  # Batch size for training
num_epochs = 20  # Number of training epochs
learning_rate = 0.001  # Learning rate

# Fetch and preprocess data from InfluxDB
client = InfluxDBClient(host='localhost', port=8086, database='your_database_name')
query = 'SELECT * FROM "your_measurement_name" WHERE time >= now() - 1h'
result = client.query(query)
data_list = list(result.get_points())

# Filter required features
filtered_features = ['tx_pkts', 'tx_error', 'cqi']
filtered_data = [[point[feature] for feature in filtered_features if feature in point] for point in data_list]

# Convert to NumPy array and reshape for RNN
data_array = np.array(filtered_data)
num_sequences = len(data_array) // seq_length
data_array = data_array[:num_sequences * seq_length].reshape(num_sequences, seq_length, len(filtered_features))
n_features = len(filtered_features)  # Update feature count

# Convert to PyTorch tensor and DataLoader
data_tensor = torch.tensor(data_array, dtype=torch.float32)
labels = torch.zeros(data_tensor.size(0))
dataset = TensorDataset(data_tensor, labels)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# RNN Autoencoder model
class RNN_Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(RNN_Autoencoder, self).__init__()
        self.encoder_rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hidden_to_latent = nn.Linear(hidden_dim, latent_dim)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_rnn = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.encoder_rnn(x)
        latent = self.hidden_to_latent(h[-1])
        h_decoded = self.latent_to_hidden(latent).unsqueeze(0)
        x_reconstructed, _ = self.decoder_rnn(x, (h_decoded, torch.zeros_like(h_decoded)))
        return x_reconstructed

# Initialize model, loss, and optimizer
model = RNN_Autoencoder(n_features, hidden_dim, latent_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_data, _ in data_loader:
        optimizer.zero_grad()
        reconstructed = model(batch_data)
        loss = criterion(reconstructed, batch_data)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Evaluate anomaly detection using reconstruction error
model.eval()
with torch.no_grad():
    reconstruction_errors = []
    for batch_data, _ in data_loader:
        reconstructed = model(batch_data)
        errors = ((batch_data - reconstructed) ** 2).mean(dim=(1, 2))
        reconstruction_errors.extend(errors.numpy())

# Example threshold for anomaly detection
threshold = 0.05
anomalies = [err > threshold for err in reconstruction_errors]
print(f"Detected {sum(anomalies)} anomalies out of {len(reconstruction_errors)} samples.")
