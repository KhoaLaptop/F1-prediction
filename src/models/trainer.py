import os
import pandas as pd
from src.models.qualifying import QualifyingModel
from src.models.sprint import SprintModel
from src.models.race import RaceModel

class ModelTrainer:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            
        self.quali_model = QualifyingModel()
        self.sprint_model = SprintModel()
        self.race_model = RaceModel()
        
    def train_qualifying(self, train_data):
        print("Training Qualifying Model...")
        X = train_data
        y = train_data['TargetPosition']
        self.quali_model.train(X, y)
        self.quali_model.save(os.path.join(self.models_dir, 'qualifying_model.pkl'))
        print("Qualifying Model trained and saved.")
        
    def train_sprint(self, train_data):
        print("Training Sprint Model...")
        X = train_data
        y = train_data['TargetPosition']
        self.sprint_model.train(X, y)
        self.sprint_model.save(os.path.join(self.models_dir, 'sprint_model.pkl'))
        print("Sprint Model trained and saved.")
        
    def train_race(self, train_data):
        print("Training Race Model...")
        X = train_data
        y = train_data['TargetPosition']
        
        # Calculate groups for ranking
        # Assuming data is sorted by Race/Round
        if 'RoundNumber' in train_data.columns:
            groups = train_data.groupby('RoundNumber').size().tolist()
        else:
            # Fallback: treat as one big group
            groups = [len(train_data)]
            
        self.race_model.train(X, y, groups)
        self.race_model.save(os.path.join(self.models_dir, 'race_model.pkl'))
        print("Race Model trained and saved.")
