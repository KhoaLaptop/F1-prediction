from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle

class SprintModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.feature_names = [
            'TrackTemp', 'OvertakeDifficulty', 
            'DriverAvgPos', 'DriverDNFRate', 'ReliabilityScore'
        ]
        self.classes = ["Top3", "Points", "NoPoints"]
        
    def _categorize_position(self, pos):
        if pos <= 3:
            return "Top3"
        elif pos <= 8:
            return "Points"
        else:
            return "NoPoints"
            
    def train(self, X, y):
        """
        Train the model.
        y: Series of numerical positions, will be converted to classes.
        """
        X_subset = X[self.feature_names].fillna(0)
        
        # Drop rows where y is NaN
        mask = y.notna()
        X_subset = X_subset[mask]
        y_clean = y[mask]
        
        y_class = y_clean.apply(self._categorize_position)
        self.model.fit(X_subset, y_class)
        
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
