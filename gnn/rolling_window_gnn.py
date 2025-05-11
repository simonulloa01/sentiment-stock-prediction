# gnn.py
import sys
import pandas as pd
import torch
from torch import nn
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data
from scipy.stats import pearsonr

# Redirect standard output to a file
output_file = open("rolling_window_output.txt", "w")
sys.stdout = output_file


def generate_rolling_temporal_graphs(df, window_size=14): # Removed target_ticker
    """
    Generate rolling temporal graphs for GNN training for the whole sector.

    Args:
        df (pd.DataFrame): The input DataFrame containing stock data.
        window_size (int): The number of days in the rolling window.

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

        # Skip if no data for current or next date (basic check)
        if df_now.empty or df_next.empty:
            continue
        
        # Get labels for ALL tickers for the next date
        labels_all_nodes_map = {row['ticker']: row['label'] for _, row in df_next.iterrows() if 'label' in row}
        y_labels_tensor = torch.zeros(num_nodes, dtype=torch.float)
        valid_labels_mask = torch.zeros(num_nodes, dtype=torch.bool)

        for node_idx, ticker_name in enumerate(unique_tickers):
            if ticker_name in labels_all_nodes_map:
                y_labels_tensor[node_idx] = labels_all_nodes_map[ticker_name]
                valid_labels_mask[node_idx] = True
            else:
                # For BCELoss, cannot use NaN. Mask will handle exclusion.
                y_labels_tensor[node_idx] = 0.5 # Placeholder, will be ignored by masked loss
        
        if not valid_labels_mask.any(): # Skip if no valid labels for any node in this graph
            print(f"Skipping graph for current date {current_date}: No valid labels for any stock on {next_date}")
            continue

        timestamp = df_now['timestamp'].values[0] # Assuming all stocks in df_now share the same timestamp for the day

        # Collect closing prices for all tickers in the rolling window
        closes = {
            ticker: df_window[df_window['ticker'] == ticker]['close'].values
            for ticker in unique_tickers
        }

        # Initialize graph components for edges
        src_edges, dst_edges, edge_attr_list = [], [], []
        
        print(f"Processing graph for current date {current_date}, predicting for {next_date}")

        # Edge generation: All-to-all with Pearson correlation
        for ticker_i_idx, ticker_i_name in enumerate(unique_tickers):
            if ticker_i_name not in closes or len(closes[ticker_i_name]) != window_size:
                continue
            for ticker_j_idx, ticker_j_name in enumerate(unique_tickers):
                if ticker_i_idx == ticker_j_idx:
                    continue
                if ticker_j_name not in closes or len(closes[ticker_j_name]) != window_size:
                    continue
                
                try:
                    corr, _ = pearsonr(closes[ticker_i_name], closes[ticker_j_name])
                    corr = 0.0 if pd.isna(corr) else corr
                except ValueError: # Handles cases with insufficient data or constant series for correlation
                    corr = 0.0
                
                src_edges.append(ticker_i_idx)
                dst_edges.append(ticker_j_idx)
                edge_attr_list.append([corr])
        
        print("----------------------------------------------")
        # Extract features for the current day, ensuring all unique_tickers are represented
        features_this_day = df_now.set_index('ticker').reindex(unique_tickers)[feature_cols].fillna(0)
        x = torch.tensor(features_this_day.values, dtype=torch.float)

        if not src_edges: # If no edges were created
            edge_index = torch.empty((2,0), dtype=torch.long)
            edge_attr = torch.empty((0,1), dtype=torch.float)
            print(f"Warning: No edges created for graph on {current_date}. Skipping this graph or using empty edges.")
            # Depending on model architecture, you might want to skip:
            # continue 
        else:
            edge_index = torch.tensor([src_edges, dst_edges], dtype=torch.long)
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

        # Create a PyTorch Geometric Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y_labels_tensor, # Labels for all nodes
            timestamp=torch.tensor([timestamp], dtype=torch.long),
            valid_labels_mask=valid_labels_mask # Mask for valid labels
        )
        all_data.append(data)

    return all_data, ticker2id


class StockPredictGNN(nn.Module): # Renamed from GOOG_GNN
    """
    A GNN model for predicting stock movements for multiple stocks (sector).
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
        return torch.sigmoid(self.classifier(x)).squeeze(-1) # Output shape: [num_nodes]


def train_gnn(model, data_list, ticker2id, epochs=5): # Removed target_ticker
    """
    Train the GNN model for the whole sector.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss(reduction='none') # Use reduction='none' to apply mask

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        processed_nodes_count = 0

        for data in data_list:
            if data.edge_index.numel() == 0 and data.x.numel() > 0 : # Skip graphs with no edges if model requires them
                 # TransformerConv might handle empty edge_index if x is not empty,
                 # but typically GNNs expect connected components or at least some edges.
                 # If your model can process nodes without edges, you might remove this check.
                print(f"Skipping training for graph at timestamp {data.timestamp.item()} due to no edges.")
                continue
            
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr) # out shape: [num_nodes]
            
            mask = data.valid_labels_mask
            if not mask.any(): # Skip if no valid labels in this graph
                continue

            masked_out = out[mask]
            masked_labels = data.y[mask]

            if masked_out.numel() == 0: # Should be caught by "if not mask.any()"
                continue
                
            loss_per_node = loss_fn(masked_out, masked_labels)
            loss = loss_per_node.mean() # Average loss over valid nodes in this graph

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf loss encountered. Skipping batch. Out: {masked_out}, Labels: {masked_labels}")
                continue

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * masked_out.numel() # Accumulate loss weighted by num valid nodes
            processed_nodes_count += masked_out.numel()

        avg_loss = total_loss / processed_nodes_count if processed_nodes_count > 0 else 0
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")


def evaluate_gnn(model, data_list, ticker2id): # Removed target_ticker
    """
    Evaluate the GNN model on the whole sector.
    """
    model.eval()
    y_true_all, y_pred_all = [], []
    id2ticker = {i: ticker for ticker, i in ticker2id.items()}


    print("\nEvaluating Sector Performance:")
    with torch.no_grad():
        for data_idx, data in enumerate(data_list):
            if data.edge_index.numel() == 0 and data.x.numel() > 0:
                print(f"Skipping evaluation for graph at index {data_idx} (timestamp {data.timestamp.item()}) due to no edges.")
                continue

            out = model(data.x, data.edge_index, data.edge_attr) # out shape: [num_nodes]
            
            mask = data.valid_labels_mask
            if not mask.any():
                continue

            masked_out = out[mask]
            masked_labels = data.y[mask]
            
            if masked_out.numel() == 0:
                continue

            pred_for_valid_nodes = (masked_out > 0.5).float()
            
            y_true_all.extend(masked_labels.tolist())
            y_pred_all.extend(pred_for_valid_nodes.tolist())

            # Optional: Print summary for the day/graph
            timestamp = data.timestamp.item()
            prediction_date_str = pd.to_datetime(timestamp, unit='s').strftime('%Y-%m-%d')
            # Example: print for a few stocks or summary
            # print(f"Graph for date (features from): {prediction_date_str} - Evaluated {masked_out.numel()} stocks")
            # for i in range(min(3, masked_out.numel())): # Print for first 3 valid stocks in this graph
            #     original_node_idx = torch.where(mask)[0][i].item() 
            #     stock_name = id2ticker.get(original_node_idx, f"Unknown_ID_{original_node_idx}")
            #     print(f"  Stock: {stock_name}, Pred: {pred_for_valid_nodes[i].item():.0f}, Actual: {masked_labels[i].item():.0f}, Prob: {masked_out[i]:.2f}")


    if not y_true_all:
        print("\nNo valid data to evaluate for the sector.")
        return

    correct = sum([1 for yt, yp in zip(y_true_all, y_pred_all) if yt == yp])
    total = len(y_true_all)
    accuracy = correct / total if total > 0 else 0
    print(f"\nOverall Sector Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    actual_down = sum([1 for yt in y_true_all if yt == 0])
    actual_up = sum([1 for yt in y_true_all if yt == 1])
    pred_down_correct = sum([1 for yt, yp in zip(y_true_all, y_pred_all) if yt == 0 and yp == 0])
    pred_up_correct = sum([1 for yt, yp in zip(y_true_all, y_pred_all) if yt == 1 and yp == 1])

    print("Class Breakdown (Overall Sector):")
    print(f"  Correct Down Predictions: {pred_down_correct} (out of {actual_down} actual down instances)")
    print(f"  Correct Up Predictions: {pred_up_correct} (out of {actual_up} actual up instances)")


# === Example usage ===
df = pd.read_csv("data/dataset.csv", sep="\t")
print(df.head())  # Ensure the DataFrame is loaded correctly

# Generate graphs for the whole sector
data, ticker2id = generate_rolling_temporal_graphs(df, window_size=30)

if not data:
    print("No graph data was generated. Check dataset and parameters.")
    sys.stdout = sys.__stdout__
    output_file.close()
    sys.exit("Exiting due to no data.")

# Initialize the model with the correct number of input features
model = StockPredictGNN(in_channels=11, edge_dim=1)  # 11 features incl. sentiment, 1 edge weight (correlation)

train_gnn(model, data, ticker2id, epochs=75) # Pass ticker2id for potential use inside train/eval if needed later
evaluate_gnn(model, data, ticker2id)

# Restore standard output to the console
sys.stdout = sys.__stdout__
output_file.close()


