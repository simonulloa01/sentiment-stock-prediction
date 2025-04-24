import os
import json
import pandas as pd
from datetime import datetime

def create_gnn_datasets(tickers, start_date, end_date, output_file, rolling_window=14):
    """
    Create train and test datasets for GNN.
    
    Args:
        tickers (list): List of company tickers.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        output_file (str): File path to save the combined dataset.
        rolling_window (int): Rolling window size for feature calculations.
    """
    data_frames = []
    
    for ticker in tickers:
        # Load price data
        price_file = f'price/raw/{ticker}.csv'
        if not os.path.exists(price_file):
            print(f"Price file for {ticker} not found.")
            continue

        price_data = pd.read_csv(price_file, sep=',', header=0, 
                                 names=['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume'])
        print(price_data.head())    
        # Parse the date column explicitly with the correct format
        price_data['date'] = pd.to_datetime(price_data['date'], format='%Y-%m-%d')

        # Filter data within the specified date range
        price_data = price_data[(price_data['date'] >= start_date) & (price_data['date'] <= end_date)]
        
        # Load sentiment data
        sentiment_folder = f'tweet/preprocessed/{ticker}'
        sentiment_data = {}
        
        if os.path.exists(sentiment_folder):
            for file in os.listdir(sentiment_folder):
                # Process all files in the folder
                date = file  # Use the file name as the date
                file_path = os.path.join(sentiment_folder, file)
                
                with open(file_path, 'r') as f:
                    tweets = [json.loads(line) for line in f]  # Load all tweets from the file
                
                # Calculate sentiment measure
                sentiment_measure = 0
                for tweet in tweets:
                    sentiment_label = tweet['sentiment']['label']
                    sentiment_score = tweet['sentiment']['score']
                    
                    if sentiment_label == 'positive':
                        sentiment_measure += 1 * sentiment_score
                    elif sentiment_label == 'negative':
                        sentiment_measure += -1 * sentiment_score
                
                # Normalize by the number of sentiments
                if len(tweets) > 0:
                    sentiment_measure /= len(tweets)
                else:
                    sentiment_measure = 0  # Default to 0 if no tweets
                
                # Store the sentiment measure for the date
                sentiment_data[date] = sentiment_measure
        else:
            print(f"Sentiment folder for {ticker} not found.")
            continue
        
        # Add ticker column
        price_data['ticker'] = ticker  

        # Reorder columns to make 'ticker' the second column
        cols = price_data.columns.tolist()
        cols.insert(1, cols.pop(cols.index('ticker')))
        price_data = price_data[cols]
        
        # Calculate WMA, SMA, lag features, number of tweets, and volatility
        price_data['wma_close'] = price_data['close'].rolling(window=rolling_window).apply(
            lambda x: sum((i + 1) * val for i, val in enumerate(x)) / sum(range(1, len(x) + 1)), raw=True
        )
        price_data['sma_close'] = price_data['close'].rolling(window=rolling_window).mean()
        price_data['lag_1_close'] = price_data['close'].shift(1)
        price_data['lag_7_close'] = price_data['close'].shift(7)
        price_data['volatility'] = price_data['close'].rolling(window=rolling_window).std()
        
        # Add sentiment to price data
        price_data['sentiment'] = price_data['date'].dt.strftime('%Y-%m-%d').map(sentiment_data).fillna(0)
        
        # Add the next day's close price
        price_data['next_close'] = price_data['close'].shift(-1)

        # Add binary label (up or down) based on the next day's close price
        price_data['label'] = (price_data['next_close'] > price_data['close']).astype(int)
        
        # Sort data by date
        price_data = price_data.sort_values(by='date')
        
        data_frames.append(price_data)
    
    # Combine data for all tickers
    combined_data = pd.concat(data_frames, ignore_index=True)
    combined_data = combined_data.sort_values(by='date')  # Ensure combined data is sorted by date
    
    # Normalize the data manually using Min-Max scaling
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'sentiment', 'adj_close',
                    'wma_close', 'sma_close', 'lag_1_close', 'lag_7_close', 'volatility']
    for col in feature_cols:
        min_val = combined_data[col].min()
        max_val = combined_data[col].max()
        combined_data[col] = (combined_data[col] - min_val) / (max_val - min_val)

    # Drop rows with missing data
    combined_data = combined_data.dropna()

    # Drop the 'next_close' column
    combined_data = combined_data.drop(columns=['next_close'])

    # Save the combined dataset to a single file
    combined_data.to_csv(output_file, index=False, sep='\t')
    
    print(f"Dataset saved to {output_file}")
    
tickers = ['AAPL', 'AMZN', 'CHL', 'CSCO', 'FB', 'GOOG', 'INTC', 'MSFT', 'ORCL', 'T', 'TSM', 'VZ']
start_date = '2014-01-01'
end_date = '2015-12-31'
output_file = 'data/dataset.csv'

create_gnn_datasets(tickers, start_date, end_date, output_file)