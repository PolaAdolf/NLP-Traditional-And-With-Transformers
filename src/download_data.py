import os
import pandas as pd
from datasets import load_dataset

def download_imdb():
    print("Downloading IMDB dataset from Hugging Face...")
    # Load IMDB dataset from Hugging Face
    dataset = load_dataset("imdb")
    
    # Convert to pandas dataframes
    print("Converting to Pandas DataFrames...")
    train_df = dataset['train'].to_pandas()
    test_df = dataset['test'].to_pandas()
    
    # Map labels: 0 -> Negative, 1 -> Positive
    label_map = {0: "Negative", 1: "Positive"}
    train_df['sentiment'] = train_df['label'].map(label_map)
    test_df['sentiment'] = test_df['label'].map(label_map)
    
    # Keep only the text and the mapped sentiment
    train_df = train_df[['text', 'sentiment']]
    test_df = test_df[['text', 'sentiment']]
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    train_file = os.path.join('data', 'imdb_train.csv')
    test_file = os.path.join('data', 'imdb_test.csv')
    
    print(f"Saving training data to {train_file}...")
    train_df.to_csv(train_file, index=False)
    
    print(f"Saving testing data to {test_file}...")
    test_df.to_csv(test_file, index=False)
    
    print("Data download and extraction complete! You are ready to start Part 1.")

if __name__ == "__main__":
    download_imdb()
