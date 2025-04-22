import os
import json
import pandas as pd
from datetime import datetime

def create_gnn_datasets(tickers, start_date, end_date, output_file):
    """
    Create train and test datasets for GNN.
    
    Args:
        tickers (list): List of company tickers.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        train_split (float): Proportion of data to use for training (e.g., 0.8 for 80%).
        train_file (str): File path to save the training dataset.
        test_file (str): File path to save the testing dataset.
    """
    data_frames = []
    
    for ticker in tickers:
        # Load price data
        price_file = f'price/preprocessed/{ticker}.txt'
        if not os.path.exists(price_file):
            print(f"Price file for {ticker} not found.")
            continue
        
        price_data = pd.read_csv(price_file, sep='\t', header=None, 
                                 names=['date', 'movement_percent', 'open', 'high', 'low', 'close', 'volume'])
        price_data['date'] = pd.to_datetime(price_data['date'])
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
                    # print(f"Loaded {len(tweets)} tweets for {ticker} on {date}.")
                
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
        
        # Add sentiment to price data
        price_data['sentiment'] = price_data['date'].dt.strftime('%Y-%m-%d').map(sentiment_data).fillna(0)
        
        # Add binary label (up or down)
        price_data['label'] = (price_data['close'] > price_data['open']).astype(int)
        
        price_data = price_data.sort_values(by='date')  # Sort data by date
        
        data_frames.append(price_data)
    
    # Combine data for all tickers
    combined_data = pd.concat(data_frames, ignore_index=True)
    combined_data = combined_data.sort_values(by='date')  # Ensure combined data is sorted by date
    
    # Save the combined dataset to a single file
    combined_data.to_csv(output_file, index=False, sep='\t')
    
    print(f"Dataset saved to {output_file}")
    
tickers = ['AAPL', 'AMZN', 'CHL', 'CSCO', 'FB', 'GOOG', 'INTC', 'MSFT', 'ORCL', 'T', 'TSM', 'VZ']
start_date = '2014-01-01'
end_date = '2015-12-31'
output_file = 'data/dataset.csv'

create_gnn_datasets(tickers, start_date, end_date, output_file)