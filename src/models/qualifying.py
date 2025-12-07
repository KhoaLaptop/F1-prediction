from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import pickle

class QualifyingModel:
    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        self.feature_names = [
            'TrackTemp', 'OvertakeDifficulty', 
            'DriverAvgPos', 'DriverDNFRate', 'QualiDeltaTeammate', 'ReliabilityScore'
        ]
        
    def train(self, X, y):
        """
        Train the model.
        X: DataFrame of features
        y: Series of target positions
        """
        # Ensure only relevant features are used
        X_subset = X[self.feature_names].fillna(0)
        
        # Drop rows where y is NaN
        mask = y.notna()
        X_subset = X_subset[mask]
        y_clean = y[mask]
        
        self.model.fit(X_subset, y_clean)
        
    def predict(self, X):
        """
        Predict qualifying position.
        """
        X_subset = X[self.feature_names].fillna(0)
        return self.model.predict(X_subset)
        
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
