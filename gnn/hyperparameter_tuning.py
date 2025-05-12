# gnn.py
import sys
import pandas as pd
import torch
from torch import nn
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Define the Tee class to duplicate output
class Tee:
    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2

    def write(self, message):
        self.stream1.write(message)
        self.stream2.write(message)
        self.stream1.flush() 
        self.stream2.flush()

    def flush(self):
        self.stream1.flush()
        self.stream2.flush()

def generate_rolling_temporal_graphs(df, window_size=14):
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
    df['timestamp'] = pd.to_datetime(df['date']).astype(int) // 10**9
    df = df.sort_values(by=['date', 'ticker'])

    unique_dates = sorted(df['date'].unique())
    unique_tickers = sorted(df['ticker'].unique())
    ticker2id = {ticker: i for i, ticker in enumerate(unique_tickers)}
    num_nodes = len(unique_tickers)

    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'wma_close', 'sma_close', 'lag_1_close', 'lag_7_close', 'volatility', 'sentiment']
    all_data = []

    for i in range(window_size, len(unique_dates) - 1):
        window_dates = unique_dates[i - window_size:i]
        current_date = unique_dates[i]
        next_date = unique_dates[i + 1]

        df_window = df[df['date'].isin(window_dates)]
        df_now = df[df['date'] == current_date]
        df_next = df[df['date'] == next_date]

        if df_now.empty or df_next.empty:
            continue
        
        labels_all_nodes_map = {row['ticker']: row['label'] for _, row in df_next.iterrows() if 'label' in row}
        y_labels_tensor = torch.zeros(num_nodes, dtype=torch.float)
        valid_labels_mask = torch.zeros(num_nodes, dtype=torch.bool)

        for node_idx, ticker_name in enumerate(unique_tickers):
            if ticker_name in labels_all_nodes_map:
                y_labels_tensor[node_idx] = labels_all_nodes_map[ticker_name]
                valid_labels_mask[node_idx] = True
            else:
                y_labels_tensor[node_idx] = 0.5 # Placeholder
        
        if not valid_labels_mask.any():
            # print(f"Skipping graph for current date {current_date}: No valid labels for any stock on {next_date}")
            continue

        timestamp = df_now['timestamp'].values[0] 

        closes = {
            ticker: df_window[df_window['ticker'] == ticker]['close'].values
            for ticker in unique_tickers
        }

        src_edges, dst_edges, edge_attr_list = [], [], []
        

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
                except ValueError: 
                    corr = 0.0
                
                src_edges.append(ticker_i_idx)
                dst_edges.append(ticker_j_idx)
                edge_attr_list.append([corr])
        
        # print("----------------------------------------------")
        features_this_day = df_now.set_index('ticker').reindex(unique_tickers)[feature_cols].fillna(0)
        x = torch.tensor(features_this_day.values, dtype=torch.float)

        if not src_edges:
            edge_index = torch.empty((2,0), dtype=torch.long)
            edge_attr = torch.empty((0,1), dtype=torch.float)
            # print(f"Warning: No edges created for graph on {current_date}.")
        else:
            edge_index = torch.tensor([src_edges, dst_edges], dtype=torch.long)
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y_labels_tensor, 
            timestamp=torch.tensor([timestamp], dtype=torch.long),
            valid_labels_mask=valid_labels_mask
        )
        all_data.append(data)

    return all_data, ticker2id


class StockPredictGNN(nn.Module): # Renamed from AAPL_GNN
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


def train_gnn(model, data_list, ticker2id, epochs=5, learning_rate=0.001, current_window_size_for_plot=None, current_lr_for_plot=None): # Removed target_ticker, added params for plotting
    """
    Train the GNN model for the whole sector.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss(reduction='none') 
    
    train_losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        processed_nodes_count = 0
        total_valid_labels_in_epoch = 0 
        graphs_processed_in_epoch = 0 

        for data in data_list:
            if data.edge_index.numel() == 0 and data.x.numel() > 0 :
                # print(f"Skipping training for graph at timestamp {data.timestamp.item()} due to no edges.")
                continue
            
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr) 
            
            mask = data.valid_labels_mask
            if not mask.any(): 
                continue

            total_valid_labels_in_epoch += mask.sum().item() 
            graphs_processed_in_epoch += 1 

            masked_out = out[mask]
            masked_labels = data.y[mask]

            if masked_out.numel() == 0: 
                continue
                
            loss_per_node = loss_fn(masked_out, masked_labels)
            loss = loss_per_node.mean() 

            if torch.isnan(loss) or torch.isinf(loss):
                # print(f"Warning: NaN or Inf loss encountered. Skipping batch. Out: {masked_out}, Labels: {masked_labels}")
                continue

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * masked_out.numel() 
            processed_nodes_count += masked_out.numel()
        
        avg_loss_epoch = total_loss / processed_nodes_count if processed_nodes_count > 0 else float('inf')
        train_losses.append(avg_loss_epoch)
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            avg_valid_labels_per_graph = total_valid_labels_in_epoch / graphs_processed_in_epoch if graphs_processed_in_epoch > 0 else 0 # New
            print(f"Epoch {epoch+1}, Avg Loss: {avg_loss_epoch:.4f}, Avg Valid Labels per Graph: {avg_valid_labels_per_graph:.2f}") # Modified

    # Plotting loss for the current hyperparameter set
    if current_window_size_for_plot is not None and current_lr_for_plot is not None:
        plt.clf()
        plt.title(f"Loss for WS={current_window_size_for_plot}, LR={current_lr_for_plot}")
        plt.plot([i + 1 for i in range(epochs)], train_losses)
        plt.xlabel("Epochs")
        plt.ylabel("Average Loss")
        plt.savefig(f"loss_ws{current_window_size_for_plot}_lr{current_lr_for_plot}.png")
        print(f"Saved loss plot to loss_ws{current_window_size_for_plot}_lr{current_lr_for_plot}.png")
    
    return train_losses[-1] if train_losses else float('inf')


def evaluate_gnn(model, data_list, ticker2id): 
    """
    Evaluate the GNN model on the whole sector.
    """
    model.eval()
    y_true_all, y_pred_all = [], []
    all_masked_outputs = []
    total_valid_labels_evaluated = 0 
    graphs_evaluated = 0
    # id2ticker = {i: ticker for ticker, i in ticker2id.items()} # For detailed per-stock printing if needed

    print("\nEvaluating Model Performance:") # Changed from "Sector Performance" for generality
    with torch.no_grad():
        for data_idx, data in enumerate(data_list):
            if data.edge_index.numel() == 0 and data.x.numel() > 0:
                # print(f"Skipping evaluation for graph at index {data_idx} (timestamp {data.timestamp.item()}) due to no edges.")
                continue

            out = model(data.x, data.edge_index, data.edge_attr) 
            
            mask = data.valid_labels_mask
            if not mask.any():
                continue
            
            total_valid_labels_evaluated += mask.sum().item() 
            graphs_evaluated += 1 

            masked_out = out[mask]
            masked_labels = data.y[mask]
            
            if masked_out.numel() == 0:
                continue

            all_masked_outputs.extend(masked_out.tolist())
            pred_for_valid_nodes = (masked_out > 0.5).float()
            
            y_true_all.extend(masked_labels.tolist())
            y_pred_all.extend(pred_for_valid_nodes.tolist())

    if not y_true_all:
        print("\nNo valid data to evaluate.") 
        return 0.0

    avg_valid_labels_per_graph_eval = total_valid_labels_evaluated / graphs_evaluated if graphs_evaluated > 0 else 0 
    print(f"Evaluation based on {len(y_true_all)} data points from {graphs_evaluated} graphs (avg {avg_valid_labels_per_graph_eval:.2f} valid labels/graph).") 

    # Analysis of raw predictions (New)
    if all_masked_outputs:
        outputs_tensor = torch.tensor(all_masked_outputs)
        print(f"  Prediction distribution (raw outputs): Min: {outputs_tensor.min():.4f}, Max: {outputs_tensor.max():.4f}, Mean: {outputs_tensor.mean():.4f}, Std: {outputs_tensor.std():.4f}")
        predictions_near_threshold = ((outputs_tensor > 0.4) & (outputs_tensor < 0.6)).sum().item()
        print(f"  Number of predictions between 0.4 and 0.6: {predictions_near_threshold} (out of {len(all_masked_outputs)})")


    correct = sum([1 for yt, yp in zip(y_true_all, y_pred_all) if yt == yp])
    total = len(y_true_all)
    accuracy = correct / total if total > 0 else 0
    print(f"\nOverall Accuracy: {accuracy:.4f} ({correct}/{total})") # Changed
    
    actual_down = sum([1 for yt in y_true_all if yt == 0])
    actual_up = sum([1 for yt in y_true_all if yt == 1])
    pred_down_correct = sum([1 for yt, yp in zip(y_true_all, y_pred_all) if yt == 0 and yp == 0])
    pred_up_correct = sum([1 for yt, yp in zip(y_true_all, y_pred_all) if yt == 1 and yp == 1])

    print("Class Breakdown (Overall Sector):")
    print(f"  Correct Down Predictions: {pred_down_correct} (out of {actual_down} actual down instances)")
    print(f"  Correct Up Predictions: {pred_up_correct} (out of {actual_up} actual up instances)")
    return accuracy


# hyperparameter tuning
output_log_file = open("hyperparameter_tuning_log.txt", "w")
original_stdout = sys.stdout
sys.stdout = Tee(original_stdout, output_log_file) 

df_full = pd.read_csv("../data/dataset.csv", sep="\\t")

window_sizes_to_tune = [90]
learning_rates_to_tune = [0.001]
epochs_for_tuning = 1000

best_accuracy = 0.0
best_params = {}
results = []

for ws in window_sizes_to_tune:
    print(f"\n--- Generating data for window_size={ws} ---")
    all_graphs, ticker_map = generate_rolling_temporal_graphs(df_full, window_size=ws) # Renamed data_graphs to all_graphs for clarity
    
    # if not data_graphs:
    if not all_graphs: # Adjusted to use all_graphs
        print(f"No data generated for window_size={ws}. Skipping.")
        continue

    # Split data into training and testing sets 80%/20%
    split_idx = int(len(all_graphs) * 0.8)
    train_graphs = all_graphs[:split_idx]
    test_graphs = all_graphs[split_idx:]

    print(f"Total graphs: {len(all_graphs)}, Training graphs: {len(train_graphs)}, Testing graphs: {len(test_graphs)}")

    if not train_graphs or not test_graphs: # Check if either is empty
        print(f"Not enough data to split for window_size={ws}. Skipping.")
        continue

    for lr in learning_rates_to_tune:
        print(f"\n--- Tuning with window_size={ws}, learning_rate={lr}, epochs={epochs_for_tuning} ---")
    
        model_tune = StockPredictGNN(in_channels=11, edge_dim=1) 
        
        # final_loss = train_gnn(model_tune, data_graphs, ticker_map, epochs=epochs_for_tuning, learning_rate=lr, current_window_size_for_plot=ws, current_lr_for_plot=lr)
        # accuracy_eval = evaluate_gnn(model_tune, data_graphs, ticker_map)
        print(f"Training on {len(train_graphs)} graphs...")
        final_loss = train_gnn(model_tune, train_graphs, ticker_map, epochs=epochs_for_tuning, learning_rate=lr, current_window_size_for_plot=ws, current_lr_for_plot=lr)
        
        print("\n--- Evaluating on Training Data (during tuning) ---")
        train_accuracy_eval = evaluate_gnn(model_tune, train_graphs, ticker_map)
        print(f"Training Accuracy (for current params): {train_accuracy_eval:.4f}")

        print("\n--- Evaluating on Test Data (during tuning) ---")
        test_accuracy_eval = evaluate_gnn(model_tune, test_graphs, ticker_map)
        print(f"Test Accuracy (for current params): {test_accuracy_eval:.4f}")
        
        # results.append({'window_size': ws, 'learning_rate': lr, 'final_loss': final_loss, 'accuracy': accuracy_eval})
        # print(f"Result: WS={ws}, LR={lr}, Final Avg Loss={final_loss:.4f}, Accuracy={accuracy_eval:.4f}")
        results.append({'window_size': ws, 'learning_rate': lr, 'final_loss': final_loss, 'train_accuracy': train_accuracy_eval, 'test_accuracy': test_accuracy_eval})
        print(f"Result: WS={ws}, LR={lr}, Final Avg Loss={final_loss:.4f}, Train Acc={train_accuracy_eval:.4f}, Test Acc={test_accuracy_eval:.4f}")

        # if accuracy_eval > best_accuracy:
        #     best_accuracy = accuracy_eval
        #     best_params = {'window_size': ws, 'learning_rate': lr}
        #     print(f"*** New best accuracy: {best_accuracy:.4f} with params: {best_params} ***")
        if test_accuracy_eval > best_accuracy: # Base best_accuracy on test set performance
            best_accuracy = test_accuracy_eval
            best_params = {'window_size': ws, 'learning_rate': lr, 'train_accuracy_at_best_test': train_accuracy_eval}
            print(f"*** New best test accuracy: {best_accuracy:.4f} with params: {best_params} ***")

print("\n--- Hyperparameter Tuning Summary ---")
for res in results:
    # print(f"WS={res['window_size']}, LR={res['learning_rate']}, Final Loss={res['final_loss']:.4f}, Accuracy={res['accuracy']:.4f}")
    print(f"WS={res['window_size']}, LR={res['learning_rate']}, Final Loss={res['final_loss']:.4f}, Train Acc={res['train_accuracy']:.4f}, Test Acc={res['test_accuracy']:.4f}")

print(f"\nBest Test Accuracy: {best_accuracy:.4f}")
print(f"Best Parameters: {{best_params}}")

print("\n--- Running with Best Parameters (or default if tuning was skipped/failed) ---")

# Use best_params found, or defaults if tuning didn't yield better ones or was skipped
final_window_size = best_params.get('window_size', 75) # Default to 75 if no best_params
final_learning_rate = best_params.get('learning_rate', 0.001) # Default to 0.001
final_epochs = 100 # Longer training with best params

print(f"Using final parameters: window_size={final_window_size}, learning_rate={final_learning_rate}, epochs={final_epochs}")

# data_final, ticker2id_final = generate_rolling_temporal_graphs(df_full, window_size=final_window_size)
all_data_final, ticker2id_final = generate_rolling_temporal_graphs(df_full, window_size=final_window_size)

# if not data_final:
if not all_data_final:
    print("No data generated for final training run. Exiting.")
    # sys.exit() # Consider exiting if no data
else:
    # Split final data into training and testing sets (80-20 split)
    final_split_idx = int(len(all_data_final) * 0.8)
    final_train_graphs = all_data_final[:final_split_idx]
    final_test_graphs = all_data_final[final_split_idx:]

    print(f"Final Run - Total graphs: {len(all_data_final)}, Training graphs: {len(final_train_graphs)}, Testing graphs: {len(final_test_graphs)}")

    if not final_train_graphs or not final_test_graphs:
        print("Not enough data for final train/test split. Exiting.")
    else:
        model_final = StockPredictGNN(in_channels=11, edge_dim=1)
        print("\n--- Training final model ---")
        # train_gnn(model_final, data_final, ticker2id_final, epochs=final_epochs, learning_rate=final_learning_rate, current_window_size_for_plot=f"final_ws{final_window_size}", current_lr_for_plot=f"final_lr{final_learning_rate}")
        # print("\n--- Evaluating final model ---")
        # evaluate_gnn(model_final, data_final, ticker2id_final)
        print(f"Training final model on {len(final_train_graphs)} graphs...")
        train_gnn(model_final, final_train_graphs, ticker2id_final, epochs=final_epochs, learning_rate=final_learning_rate, current_window_size_for_plot=f"final_ws{final_window_size}", current_lr_for_plot=f"final_lr{final_learning_rate}")
        
        print("\n--- Evaluating final model on Training Data ---")
        final_train_accuracy = evaluate_gnn(model_final, final_train_graphs, ticker2id_final)
        print(f"Final Model Training Accuracy: {final_train_accuracy:.4f}")

        print("\n--- Evaluating final model on Test Data ---")
        final_test_accuracy = evaluate_gnn(model_final, final_test_graphs, ticker2id_final)
        print(f"Final Model Test Accuracy: {final_test_accuracy:.4f}")


print("\\nHyperparameter tuning script finished.")
#idk why we need this but it reset the stdout state for next run
if isinstance(sys.stdout, Tee): 
    sys.stdout = sys.stdout.stream1 
output_log_file.close()