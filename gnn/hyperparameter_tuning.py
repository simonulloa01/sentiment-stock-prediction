# gnn.py
import sys
import pandas as pd
import torch
from torch import nn
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def generate_rolling_temporal_graphs(df, window_size=14, target_ticker="AAPL"):
    """
    Generate rolling temporal graphs for GNN training.

    Args:
        df (pd.DataFrame): The input DataFrame containing stock data.
        window_size (int): The number of days in the rolling window.
        target_ticker (str): The ticker for which predictions are made.

    Returns:
        list: A list of PyTorch Geometric Data objects representing the graphs.
        dict: A mapping of tickers to node indices.
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['date']).astype(int) // 10**9  # Convert date to Unix timestamp
    df = df.sort_values(by=['date', 'ticker'])  # Ensure data is sorted by date and ticker

    unique_dates = sorted(df['date'].unique())  # Get all unique dates
    unique_tickers = sorted(df['ticker'].unique())  # Get all unique tickers
    ticker2id = {ticker: i for i, ticker in enumerate(unique_tickers)}  # Map tickers to node indices
    num_nodes = len(unique_tickers)  # Total number of nodes (tickers)

    # Ensure feature columns match the dataset
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'wma_close', 'sma_close', 'lag_1_close', 'lag_7_close', 'volatility', 'sentiment']  # Features for each node
    all_data = []  # List to store graph data

    for i in range(window_size, len(unique_dates) - 1):
        # Define the rolling window and the current/next dates
        window_dates = unique_dates[i - window_size:i]
        current_date = unique_dates[i]
        next_date = unique_dates[i + 1]

        # Filter data for the rolling window, current date, and next date
        df_window = df[df['date'].isin(window_dates)]
        df_now = df[df['date'] == current_date]
        df_next = df[df['date'] == next_date]

        # Skip if the target ticker is not present in the current or next date
        if target_ticker not in df_now['ticker'].values or target_ticker not in df_next['ticker'].values:
            continue

        # Get the label for the target ticker (up or down)
        label = df_next[df_next['ticker'] == target_ticker]['label'].values[0]
        timestamp = df_now['timestamp'].values[0]

        # Collect closing prices for all tickers in the rolling window
        closes = {
            ticker: df_window[df_window['ticker'] == ticker]['close'].values
            for ticker in unique_tickers
        }

        # Get the closing prices for the target ticker
        target_closes = closes[target_ticker]
        if len(target_closes) != window_size:
            continue

        # Initialize graph components
        src, dst, t, msg, y = [], [], [], [], []

        # Print the rolling window and prediction date
        # print(f"Prediction for {next_date}, using features from current date {current_date} and past window {window_dates[0]} to {window_dates[-1]}")

        for ticker, close_vals in closes.items():
            if ticker == target_ticker or len(close_vals) != window_size:
                continue
            # Calculate Pearson correlation between the target ticker and other tickers
            corr, _ = pearsonr(close_vals, target_closes)
            corr = 0.0 if pd.isna(corr) else corr

            # Print the correlation for debugging
            # print(f"Correlation between {target_ticker} and {ticker}: {corr:.4f}")

            # Add edge information
            src.append(ticker2id[ticker])
            dst.append(ticker2id[target_ticker])
            t.append(timestamp)
            msg.append([corr])
            y.append(label)
        # print("----------------------------------------------")
        # Extract features for the current day
        features_this_day = df_now.set_index('ticker').reindex(unique_tickers)[feature_cols].fillna(0)
        x = torch.tensor(features_this_day.values, dtype=torch.float)

        # Create a PyTorch Geometric Data object
        data = Data(
            x=x,
            edge_index=torch.tensor([src, dst], dtype=torch.long),
            edge_attr=torch.tensor(msg, dtype=torch.float),
            y=torch.tensor(y, dtype=torch.float),
            timestamp=torch.tensor([timestamp], dtype=torch.long)
        )
        all_data.append(data)

    return all_data, ticker2id


class AAPL_GNN(nn.Module):
    """
    A simple GNN model for predicting stock movements for AAPL.
    """
    def __init__(self, in_channels, edge_dim):
        super().__init__()
        self.conv1 = TransformerConv(in_channels=in_channels, out_channels=64, edge_dim=edge_dim)
        self.conv2 = TransformerConv(in_channels=64, out_channels=32, edge_dim=edge_dim)
        self.classifier = nn.Linear(32, 1)

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass of the GNN model.
        """
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.conv2(x, edge_index, edge_attr).relu()
        return torch.sigmoid(self.classifier(x)).squeeze()


def train_gnn(model, data_list, ticker2id, target_ticker="AAPL", epochs=5, learning_rate=0.001):
    """
    Train the GNN model.

    Args:
        model (nn.Module): The GNN model.
        data_list (list): List of graph data objects.
        ticker2id (dict): Mapping of tickers to node indices.
        target_ticker (str): The target ticker for predictions.
        epochs (int): Number of training epochs.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss()
    target_idx = ticker2id[target_ticker]
    train_losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for data in data_list:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr)
            pred = out[target_idx]
            label = data.y[0]
            loss = loss_fn(pred.unsqueeze(0), label.unsqueeze(0))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss)
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    plt.clf()
    plt.title("Loss Graph for GNN")
    plt.plot([i for i in range(epochs)], train_losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(f"loss.png")


def evaluate_gnn(model, data_list, ticker2id, target_ticker="AAPL"):
    """
    Evaluate the GNN model.

    Args:
        model (nn.Module): The trained GNN model.
        data_list (list): List of graph data objects.
        ticker2id (dict): Mapping of tickers to node indices.
        target_ticker (str): The target ticker for predictions.
    """
    model.eval()
    target_idx = ticker2id[target_ticker]

    y_true, y_pred = [], []

    print("\nPredictions for AAPL:")
    with torch.no_grad():
        for data in data_list:
            out = model(data.x, data.edge_index, data.edge_attr)
            pred = (out[target_idx] > 0.5).float().item()
            label = data.y[0].item()
            y_true.append(label)
            y_pred.append(pred)

             # Convert timestamp to YYYY-MM-DD format
            timestamp = timestamp = data.timestamp.item()
            prediction_date = pd.to_datetime(timestamp, unit='s').strftime('%Y-%m-%d')

            # Print the prediction for each day
            print(f"Date: {prediction_date}, Prediction: {pred:.4f}, Actual: {label}")

    correct = sum([1 for yt, yp in zip(y_true, y_pred) if yt == yp])
    total = len(y_true)
    accuracy = correct / total if total > 0 else 0
    print(f"\nAccuracy: {accuracy:.4f} ({correct}/{total})")
    print("Class Breakdown:")
    down = sum([1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0])
    up = sum([1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1])
    print(f"Correct Down: {down}, Correct Up: {up}")


# hyperparameter tuning

# window_sizes = [15, 30, 45, 60, 75, 90]
# learning_rates = [0.001, 0.01, 0.1]
# for window_size in window_sizes:
#     for learning_rate in learning_rates:
#         print(f"\nRunning with window_size={window_size} and learning_rate={learning_rate}")
#         df = pd.read_csv("../data/dataset.csv", sep="\t")
#         data, ticker2id = generate_rolling_temporal_graphs(df, window_size=window_size, target_ticker="AAPL")
#         # Initialize the model with the correct number of input features
#         model = AAPL_GNN(in_channels=11, edge_dim=1)  # 11 features incl. sentiment, 1 edge weight
#         train_gnn(model, data, ticker2id, epochs=100, window_size=window_size, learning_rate=learning_rate)
# window_size=75 and learning_rate=0.001 minimized loss to 213.9834 for 100 epochs

# window_sizes = [i for i in range(70, 81)]
# for window_size in window_sizes:
#     print(f"\nRunning with window_size={window_size}")
#     df = pd.read_csv("../data/dataset.csv", sep="\t")
#     data, ticker2id = generate_rolling_temporal_graphs(df, window_size=window_size, target_ticker="AAPL")
#     # Initialize the model with the correct number of input features
#     model = AAPL_GNN(in_channels=11, edge_dim=1)  # 11 features incl. sentiment, 1 edge weight
#     train_gnn(model, data, ticker2id, epochs=100, window_size=window_size, learning_rate=0.001)
# min loss 224.2325 over 100 epochs for window size = 75 for range(70,81)

df = pd.read_csv("../data/dataset.csv", sep="\t")
data, ticker2id = generate_rolling_temporal_graphs(df, window_size=75, target_ticker="AAPL")
model = AAPL_GNN(in_channels=11, edge_dim=1)
train_gnn(model, data, ticker2id, epochs=200, learning_rate=0.001)
evaluate_gnn(model, data, ticker2id)

# plt.clf()
# plt.title("Accuracy vs Epochs")
# plt.plot([100,150,175,200,300,500], [0.7029,0.7271,0.7802,0.8164,0.8188,0.8164])
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.savefig(f"acc_vs_epochs.png")