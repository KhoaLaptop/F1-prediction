import sys
import os
import pandas as pd
import fastf1

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import get_session_data
from src.features.processor import extract_features_for_session

# Enable cache for test
fastf1.Cache.enable_cache('fastf1_cache')

def test_feature_extraction():
    print("Testing feature extraction...")
    
    # Load Monaco 2023 Q
    monaco_q = get_session_data(2023, 'Monaco', 'Q')
    
    # Create dummy history data
    # We need columns: Driver, RoundNumber, Position, Status, TeamName
    # Let's just use the results from Monaco itself as "history" for simplicity of the test structure,
    # but logically we should use previous races.
    # For the test, we just want to see if the code runs.
    
    # Let's load Miami 2023 as history
    miami_r = get_session_data(2023, 'Miami', 'R')
    
    history_data = []
    if miami_r and miami_r.results is not None:
        for driver, row in miami_r.results.iterrows():
            history_data.append({
                'Driver': driver,
                'RoundNumber': miami_r.event['RoundNumber'],
                'Position': row['Position'],
                'Status': row['Status'],
                'TeamName': row['TeamName']
            })
            
    history_df = pd.DataFrame(history_data)
    
    print(f"History data shape: {history_df.shape}")
    
    features = extract_features_for_session(monaco_q, history_df, history_df)
    
    if not features.empty:
        print("Successfully extracted features.")
        print(features.head())
        print("Columns:", features.columns.tolist())
    else:
        print("Failed to extract features.")

if __name__ == "__main__":
    test_feature_extraction()
