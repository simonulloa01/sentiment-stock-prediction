# gnn.py
import pandas as pd
import torch
from torch import nn
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data
from scipy.stats import pearsonr


def generate_rolling_temporal_graphs(df, window_size=14, target_ticker="AAPL"):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['date']).astype(int) // 10**9
    df = df.sort_values(by=['date', 'ticker'])

    unique_dates = sorted(df['date'].unique())
    unique_tickers = sorted(df['ticker'].unique())
    ticker2id = {ticker: i for i, ticker in enumerate(unique_tickers)}
    num_nodes = len(unique_tickers)

    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'sentiment']
    all_data = []

    for i in range(window_size, len(unique_dates) - 1):
        window_dates = unique_dates[i - window_size:i]
        current_date = unique_dates[i]
        next_date = unique_dates[i + 1]

        df_window = df[df['date'].isin(window_dates)]
        df_now = df[df['date'] == current_date]
        df_next = df[df['date'] == next_date]

        if target_ticker not in df_now['ticker'].values or target_ticker not in df_next['ticker'].values:
            continue

        label = df_next[df_next['ticker'] == target_ticker]['label'].values[0]
        timestamp = df_now['timestamp'].values[0]

        closes = {
            ticker: df_window[df_window['ticker'] == ticker]['close'].values
            for ticker in unique_tickers
        }

        target_closes = closes[target_ticker]
        if len(target_closes) != window_size:
            continue

        src, dst, t, msg, y = [], [], [], [], []

        for ticker, close_vals in closes.items():
            if ticker == target_ticker or len(close_vals) != window_size:
                continue
            corr, _ = pearsonr(close_vals, target_closes)
            corr = 0.0 if pd.isna(corr) else corr

            src.append(ticker2id[ticker])
            dst.append(ticker2id[target_ticker])
            t.append(timestamp)
            msg.append([corr])
            y.append(label)

        features_this_day = df_now.set_index('ticker').reindex(unique_tickers)[feature_cols].fillna(0)
        x = torch.tensor(features_this_day.values, dtype=torch.float)

        data = Data(
            x=x,
            edge_index=torch.tensor([src, dst], dtype=torch.long),
            edge_attr=torch.tensor(msg, dtype=torch.float),
            y=torch.tensor(y, dtype=torch.float)
        )
        all_data.append(data)

    return all_data, ticker2id


class AAPL_GNN(nn.Module):
    def __init__(self, in_channels, edge_dim):
        super().__init__()
        self.conv1 = TransformerConv(in_channels=in_channels, out_channels=64, edge_dim=edge_dim)
        self.conv2 = TransformerConv(in_channels=64, out_channels=32, edge_dim=edge_dim)
        self.classifier = nn.Linear(32, 1)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.conv2(x, edge_index, edge_attr).relu()
        return torch.sigmoid(self.classifier(x)).squeeze()


def train_gnn(model, data_list, ticker2id, target_ticker="AAPL", epochs=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()
    target_idx = ticker2id[target_ticker]

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

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


def evaluate_gnn(model, data_list, ticker2id, target_ticker="AAPL"):
    model.eval()
    target_idx = ticker2id[target_ticker]

    y_true, y_pred = [], []

    with torch.no_grad():
        for data in data_list:
            out = model(data.x, data.edge_index, data.edge_attr)
            pred = (out[target_idx] > 0.5).float().item()
            label = data.y[0].item()
            y_true.append(label)
            y_pred.append(pred)

    correct = sum([1 for yt, yp in zip(y_true, y_pred) if yt == yp])
    total = len(y_true)
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print("Class Breakdown:")
    down = sum([1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0])
    up = sum([1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1])
    print(f"Correct Down: {down}, Correct Up: {up}")


# === Example usage ===
df = pd.read_csv("data/dataset.csv", sep="\t")
print(df.head())  # Ensure the DataFrame is loaded correctly
data, ticker2id = generate_rolling_temporal_graphs(df, window_size=30, target_ticker="AAPL")
model = AAPL_GNN(in_channels=6, edge_dim=1)  # 6 features incl. sentiment, 1 edge weight
train_gnn(model, data, ticker2id, epochs=50)
evaluate_gnn(model, data, ticker2id)


