import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.trainer import ModelTrainer

def test_model_training():
    print("Testing model training with synthetic data...")
    
    # Generate synthetic data
    n_samples = 100
    data = pd.DataFrame({
        'TrackTemp': np.random.uniform(20, 50, n_samples),
        'OvertakeDifficulty': np.random.randint(1, 10, n_samples),
        'DriverAvgPos': np.random.uniform(1, 20, n_samples),
        'DriverDNFRate': np.random.uniform(0, 0.5, n_samples),
        'QualiDeltaTeammate': np.random.uniform(-1, 1, n_samples),
        'ReliabilityScore': np.random.uniform(0.5, 1.0, n_samples),
        'TargetPosition': np.random.randint(1, 20, n_samples),
        'RoundNumber': np.random.choice([1, 2, 3, 4, 5], n_samples) # For grouping
    })
    
    # Sort by RoundNumber for Ranker
    data = data.sort_values('RoundNumber')
    
    trainer = ModelTrainer(models_dir='test_models')
    
    try:
        trainer.train_qualifying(data)
        trainer.train_sprint(data)
        trainer.train_race(data)
        print("All models trained successfully.")
    except Exception as e:
        print(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    test_model_training()
