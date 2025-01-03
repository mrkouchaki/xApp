import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define parameters
n_features = 6  # Number of features in the input data (can be updated later)
seq_length = 10  # Number of time steps in the sequence
hidden_dim = 64  # Hidden dimension for the RNN
latent_dim = 32  # Dimension of the latent space
batch_size = 32  # Batch size for training
num_epochs = 20  # Number of training epochs
learning_rate = 0.001  # Learning rate

# RNN Autoencoder model
class RNN_Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(RNN_Autoencoder, self).__init__()
        # Encoder
        self.encoder_rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hidden_to_latent = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_rnn = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        # Encode
        _, (h, _) = self.encoder_rnn(x)
        latent = self.hidden_to_latent(h[-1])  # Use the last hidden state
        # Decode
        h_decoded = self.latent_to_hidden(latent).unsqueeze(0)
        x_reconstructed, _ = self.decoder_rnn(x, (h_decoded, torch.zeros_like(h_decoded)))
        return x_reconstructed

# Load sample data (dummy data for illustration)
torch.manual_seed(42)
data = torch.rand((1000, seq_length, n_features))  # 1000 samples, seq_length, and n_features
labels = torch.zeros(1000)  # Dummy labels (for evaluation only)

# Create DataLoader
dataset = TensorDataset(data, labels)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
        errors = ((batch_data - reconstructed) ** 2).mean(dim=(1, 2))  # MSE per sample
        reconstruction_errors.extend(errors.numpy())

# Example threshold for anomaly detection
threshold = 0.05  # Set based on validation data or domain knowledge
anomalies = [err > threshold for err in reconstruction_errors]
print(f"Detected {sum(anomalies)} anomalies out of {len(reconstruction_errors)} samples.")
