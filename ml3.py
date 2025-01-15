import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from influxdb import InfluxDBClient
import numpy as np
from datetime import datetime, timedelta

def run_autoencoder_influxdb():
    # Define parameters
    seq_length = 10 
    hidden_dim = 64
    latent_dim = 32
    batch_size = 32 
    num_epochs = 1
    learning_rate = 0.001
    fetch_interval = 10  # Fetch new data every 10 seconds
    initial_training_duration = timedelta(hours=1)  # Training phase duration

    # Fetch and preprocess data from InfluxDB
    client = InfluxDBClient(host='localhost', port=8086, database='your_database_name')

    # Define time window for initial training
    current_time = datetime.utcnow()
    start_time = current_time - initial_training_duration  # Start fetching from 1 hour ago

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
    n_features = 3  # Adjust based on the number of features (e.g., tx_pkts, tx_error, cqi)
    model = RNN_Autoencoder(input_dim=n_features, hidden_dim=hidden_dim, latent_dim=latent_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ---- 1. TRAINING PHASE ---- #
    print("Starting initial training phase...")
    query = f'''
        SELECT "tx_pkts", "tx_error", "cqi"
        FROM "your_measurement_name"
        WHERE time >= '{start_time.isoformat()}Z' AND time < '{current_time.isoformat()}Z'
        ORDER BY time ASC
    '''
    result = client.query(query)
    data_list = list(result.get_points())

    if not data_list:
        print("No data available for initial training. Exiting...")
        return

    # Extract and preprocess data
    data_values = [
        [point.get('tx_pkts', 0), point.get('tx_error', 0), point.get('cqi', 0)]
        for point in data_list
    ]
    data_array = np.array(data_values, dtype=np.float32)

    # Check if enough data is available for a full sequence
    if data_array.size == 0:
        print("No data points available for conversion to tensor.")
        return

    print(f"Data array shape before reshaping: {data_array.shape}")
    if len(data_array) < seq_length:
        print("Not enough data points for a full sequence during training. Exiting...")
        return

    print(f"Data array dtype: {data_array.dtype}")

    # Reshape into sequences for RNN
    num_sequences = len(data_array) // seq_length
    data_array = data_array[:num_sequences * seq_length].reshape(num_sequences, seq_length, n_features)
    print(f"Reshaped data array shape: {data_array.shape}")
    print("Sample data (first sequence):")
    print(data_array[0])

    try:
        print('inside the try -------')
        data_tensor = torch.from_numpy(data_array)
        print(f"Data tensor created with shape: {data_tensor.shape}")
    except Exception as e:
        print(f"Error converting to tensor: {e}")
        return

    # DataLoader preparation
    labels = torch.zeros(data_tensor.size(0))
    print('labels:', labels)
    dataset = TensorDataset(data_tensor, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Train the model
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_data, _ in data_loader:
            print(f"Batch data shape: {batch_data.shape}")  # Should be [batch_size, seq_length, n_features]
            if batch_data.shape[-1] != n_features:
                raise ValueError(f"Input dimension mismatch! Expected last dimension to be {n_features}, but got {batch_data.shape[-1]}.")
            
            optimizer.zero_grad()
            reconstructed = model(batch_data)
            print(f"Reconstructed data shape: {reconstructed.shape}")  # Should match batch_data.shape

            loss = criterion(reconstructed, batch_data)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Training completed for epoch {epoch + 1}. Loss: {epoch_loss:.4f}")

    print("Initial training completed. Switching to evaluation mode...")

    # ---- 2. EVALUATION/INFERENCE PHASE ---- #
    while True:
        print(f"Fetching new data for anomaly detection from {current_time} to present...")
        start_time = current_time
        current_time = datetime.utcnow()
        query = f'''
            SELECT "tx_pkts", "tx_error", "cqi"
            FROM "your_measurement_name"
            WHERE time >= '{start_time.isoformat()}Z' AND time < '{current_time.isoformat()}Z'
            ORDER BY time ASC
        '''
        result = client.query(query)
        data_list = list(result.get_points())

        if not data_list:
            print("No new data available. Waiting for the next fetch interval...")
            time.sleep(fetch_interval)
            continue

        # Extract and preprocess data
        data_values = [
            [point.get('tx_pkts', 0), point.get('tx_error', 0), point.get('cqi', 0)]
            for point in data_list
        ]
        data_array = np.array(data_values, dtype=np.float32)

        if len(data_array) < seq_length:
            print("Not enough data points for a full sequence.")
            continue

        # Reshape into sequences
        num_sequences = len(data_array) // seq_length
        data_array = data_array[:num_sequences * seq_length].reshape(num_sequences, seq_length, n_features)
        data_tensor = torch.from_numpy(data_array)
        labels = torch.zeros(data_tensor.size(0))
        dataset = TensorDataset(data_tensor, labels)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Anomaly detection
        model.eval()
        with torch.no_grad():
            reconstruction_errors = []
            for i, (batch_data, _) in enumerate(data_loader):
                reconstructed = model(batch_data)
                errors = ((batch_data - reconstructed) ** 2).mean(dim=(1, 2)).numpy()

                for seq_idx, error in enumerate(errors):
                    probability = (error / threshold) * 100
                    if error > threshold:
                        print(f"Sequence {i * batch_size + seq_idx + 1}: Anomaly detected with probability {probability:.2f}%.")
                    else:
                        print(f"Sequence {i * batch_size + seq_idx + 1}: Normal data with low reconstruction error ({probability:.2f}%).")

        time.sleep(fetch_interval)

# Entry point for standalone execution
if __name__ == "__main__":
    run_autoencoder_influxdb()
