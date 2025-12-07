import xgboost as xgb
import pandas as pd
import pickle
import numpy as np

class RaceModel:
    def __init__(self):
        # XGBoost Ranker requires group info for ranking tasks
        # For simplicity in this implementation, we might use XGBRegressor or Classifier if Ranker is too complex for the demo data
        # But user requested XGBoostRanker.
        self.model = xgb.XGBRanker(
            tree_method="hist",
            objective="rank:pairwise",
            learning_rate=0.1,
            n_estimators=100
        )
        self.feature_names = [
            'TrackTemp', 'OvertakeDifficulty',
            'DriverAvgPos', 'DriverDNFRate', 'ReliabilityScore', 'QualiDeltaTeammate',
            'GridPosition', 'RacePace', 'TireDegradation', 'TopSpeed', 'RainProbability'
        ]

    def train(self, X, y, groups):
        """
        Train the model.
        groups: List of integers representing number of items in each query (race).
        """
        X_subset = X[self.feature_names].fillna(0)
        # XGBRanker expects y to be relevance scores. Lower position = higher relevance.
        # We can invert position: e.g. 21 - position. 
        # Handle NaNs in position (e.g. DNF) by filling with 21 (so relevance becomes 0)
        y_filled = y.fillna(21)
        y_relevance = 21 - y_filled
        # Clip negative values just in case
        y_relevance = y_relevance.clip(lower=0)
        
        self.model.fit(X_subset, y_relevance, group=groups)
        
    def predict(self, X):
        X_subset = X[self.feature_names].fillna(0)
        return self.model.predict(X_subset)
        
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
