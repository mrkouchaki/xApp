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

    # Fetch and preprocess data from InfluxDB
    client = InfluxDBClient(host='localhost', port=8086, database='your_database_name')

    # Define time window for fetching data
    current_time = datetime.utcnow()
    start_time = current_time - timedelta(hours=1)  # Start fetching from 1 hour ago

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
    model = RNN_Autoencoder(n_features, hidden_dim, latent_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Start time loop
    while True:
        print(f"Fetching data from {start_time} to {current_time}...")
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
            current_time = datetime.utcnow()
            continue

        # Extract and preprocess data
        data_values = [
            [point.get('tx_pkts', 0), point.get('tx_error', 0), point.get('cqi', 0)]
            for point in data_list
        ]
        data_array = np.array(data_values, dtype=np.float32)

        #data_array Might Be Empty
        if data_array.size == 0:
            print("No data points available for conversion to tensor.")
            continue 

        #data_array Shape Issues
        print(f"Data array shape before reshaping: {data_array.shape}")
        if len(data_array) < seq_length:
            print("Not enough data points for a full sequence.")
            continue
            
        # Dtype should be 'np.float32'
        print(f"Data array dtype: {data_array.dtype}")
                
        # Reshape into sequences for RNN
        num_sequences = len(data_array) // seq_length
        data_array = data_array[:num_sequences * seq_length].reshape(num_sequences, seq_length, n_features)

        #--------------------------------------------------

        print(f"Reshaped data array shape: {data_array.shape}")

        print("Sample data (first sequence):")
        print(data_array[0])

        try:
            print('inside the try -------')
            data_tensor = torch.from_numpy(data_array)
            print(f"Data tensor created with shape: {data_tensor.shape}")
        except Exception as e:
            print(f"Error converting to tensor: {e}")
            continue

        # DataLoader preparation
        labels = torch.zeros(data_tensor.size(0))
        dataset = TensorDataset(data_tensor, labels)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        #------------------------------------

        # Train the model
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
            print(f"Training completed for current batch. Loss: {epoch_loss:.4f}")

        # Evaluate anomaly detection using reconstruction error
        model.eval()
        with torch.no_grad():
            reconstruction_errors = []
            for batch_data, _ in data_loader:
                reconstructed = model(batch_data)
                errors = ((batch_data - reconstructed) ** 2).mean(dim=(1, 2))
                reconstruction_errors.extend(errors.numpy())

        # Detect anomalies
        threshold = 0.05  # Example threshold
        anomalies = [err > threshold for err in reconstruction_errors]
        print(f"Detected {sum(anomalies)} anomalies out of {len(reconstruction_errors)} samples.")

        # Update time window
        start_time = current_time
        current_time = datetime.utcnow()
        time.sleep(fetch_interval)

# Entry point for standalone execution
if __name__ == "__main__":
    run_autoencoder_influxdb()
