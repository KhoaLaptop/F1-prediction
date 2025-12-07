import argparse
import os
import pandas as pd
from src.data.loader import load_all_season_sessions
from src.features.processor import extract_features_for_session
from src.models.trainer import ModelTrainer

def train_on_real_data(seasons, models_dir, output_data=None, input_data=None):
    
    full_dataset = pd.DataFrame()

    # 1. Load Data (from file or fresh download)
    if input_data:
        print(f"Loading training data from {input_data}...")
        for file in input_data:
            if os.path.exists(file):
                df = pd.read_csv(file)
                full_dataset = pd.concat([full_dataset, df])
            else:
                print(f"Warning: File {file} not found.")
        print(f"Loaded {len(full_dataset)} rows of data.")
        
    else:
        print(f"Starting data ingestion pipeline for seasons: {seasons}")
        all_features = []
        
        # We need a global history dataframe that grows.
        driver_history = pd.DataFrame(columns=['Driver', 'RoundNumber', 'Position', 'Status', 'TeamName'])
        constructor_history = pd.DataFrame(columns=['TeamName', 'RoundNumber', 'Position', 'Status'])
        
        for year in seasons:
            print(f"Processing Season {year}...")
            # Load FP2 for Race Pace (or FP1 if Sprint weekend)
            # We need to know if it's a Sprint weekend.
            # FastF1 schedule has 'EventFormat' usually? Or we check sessions.
            # Let's load 'FP2' and 'FP1' if available.
            sessions = load_all_season_sessions(year, session_types=['Q', 'S', 'R', 'FP1', 'FP2'])
            
            for event in sessions:
                round_num = event['round']
                event_name = event['event_name']
                
                # Determine practice session for race pace
                # Default to FP2. If Sprint weekend, FP2 might be parc ferme or useless?
                # Actually in Sprint format: FP1 -> Quali -> Sprint Shootout -> Sprint -> Race.
                # FP1 is the only representative practice.
                # In Conventional: FP1 -> FP2 -> FP3 -> Quali -> Race. FP2 is usually race sims.
                
                practice_session = None
                if 'FP2' in event['sessions']:
                    practice_session = event['sessions']['FP2']
                elif 'FP1' in event['sessions']:
                    # Fallback to FP1 if FP2 missing (e.g. rain or sprint weekend where we didn't load FP2?)
                    # Actually for Sprint weekends, we should prefer FP1?
                    # Let's just use FP2 if available, else FP1.
                    practice_session = event['sessions']['FP1']
                    
                if practice_session:
                     # Ensure loaded
                     if getattr(practice_session, '_laps', None) is None:
                        try:
                            print(f"  Loading practice data for {event_name}...")
                            practice_session.load(telemetry=False, weather=False) # Telemetry needed for laps? Yes.
                        except Exception:
                            pass

                # ... (Quali and Sprint processing remains similar but we don't use practice for them currently)
                
                # Process Qualifying
                if 'Q' in event['sessions']:
                    session = event['sessions']['Q']
                    print(f"  Extracting features for {event_name} Qualifying...")
                    # Ensure data is loaded
                    if getattr(session, '_laps', None) is None:
                        print("    Data not loaded, attempting to load...")
                        try:
                            session.load()
                        except Exception as e:
                            print(f"    Failed to load session: {e}")
                            continue
                    
                    # Verify laps are actually available
                    try:
                        _ = session.laps
                    except Exception:
                        print("    Laps data not available even after load. Skipping.")
                        continue
                            
                    feats = extract_features_for_session(session, driver_history, constructor_history)
                    feats['SessionType'] = 'Qualifying'
                    all_features.append(feats)
                    
                # Process Sprint
                if 'S' in event['sessions']:
                    session = event['sessions']['S']
                    print(f"  Extracting features for {event_name} Sprint...")
                    # Ensure data is loaded
                    if getattr(session, '_laps', None) is None:
                        print("    Data not loaded, attempting to load...")
                        try:
                            session.load()
                        except Exception as e:
                            print(f"    Failed to load session: {e}")
                            continue
                    
                    # Verify laps are actually available
                    try:
                        _ = session.laps
                    except Exception:
                        print("    Laps data not available even after load. Skipping.")
                        continue
                            
                    feats = extract_features_for_session(session, driver_history, constructor_history)
                    feats['SessionType'] = 'Sprint'
                    all_features.append(feats)
                    
                    # Update history with Sprint results? usually stats exclude sprint or count differently.
                    # Let's ignore sprint for history stats for now to keep it simple, or add it.
                    
                # Process Race
                if 'R' in event['sessions']:
                    session = event['sessions']['R']
                    print(f"  Extracting features for {event_name} Race...")
                    # Ensure data is loaded
                    if getattr(session, '_laps', None) is None:
                        print("    Data not loaded, attempting to load...")
                        try:
                            session.load()
                        except Exception as e:
                            print(f"    Failed to load session: {e}")
                            continue
                    
                    # Verify laps are actually available
                    try:
                        _ = session.laps
                    except Exception:
                        print("    Laps data not available even after load. Skipping.")
                        continue
                            
                    # Pass practice session for Race Pace
                    feats = extract_features_for_session(session, driver_history, constructor_history, practice_session=practice_session)
                    feats['SessionType'] = 'Race'
                    all_features.append(feats)
                    
                    # Update History
                    if session.results is not None:
                        # Append new results to history
                        # We need to format it to match our expected history columns
                        # session.results has 'Abbreviation', 'TeamName', 'Position', 'Status'
                        
                        # Driver History
                        new_d_hist = session.results[['Abbreviation', 'Position', 'Status', 'TeamName']].copy()
                        new_d_hist.columns = ['Driver', 'Position', 'Status', 'TeamName'] # Rename Abbreviation to Driver
                        new_d_hist['RoundNumber'] = round_num # This is round within season. 
                        # Ideally history should have unique ID or Year+Round.
                        # Our feature calc uses "RoundNumber < current". If we mix years, we need Year column.
                        new_d_hist['Year'] = year
                        
                        # We need to update our feature extractors to handle Year if we train on multiple years.
                        # For now, let's assume we reset history each year or handle it.
                        # Actually, the feature extractors in src/features/driver.py use 'RoundNumber'.
                        # If we pass multi-year history, we need to filter by year or have a global counter.
                        # Let's just keep history for the current year for simplicity of this demo script.
                        
                        driver_history = pd.concat([driver_history, new_d_hist])
                        
                        # Constructor History
                        # We can aggregate by team
                        # ... (simplified for now, just using driver rows as proxy if needed or same df)
                        constructor_history = pd.concat([constructor_history, new_d_hist]) # Reusing same structure
                        
            # Reset history for next season? 
            # Usually stats carry over (e.g. last 5 races could span seasons).
            # But our 'RoundNumber' logic in features assumes 1..23.
            # We would need to improve feature engineering to handle cross-season rolling windows.
            # For this MVP, we'll clear history to avoid Round 1 2024 seeing Round 22 2023 as "future" if we just concat.
            driver_history = pd.DataFrame(columns=['Driver', 'RoundNumber', 'Position', 'Status', 'TeamName'])
            constructor_history = pd.DataFrame(columns=['TeamName', 'RoundNumber', 'Position', 'Status'])
    
            if all_features:
                full_dataset = pd.concat(all_features)
                
                if output_data:
                    print(f"Saving extracted features to {output_data}...")
                    # Create dir if needed
                    out_dir = os.path.dirname(output_data)
                    if out_dir and not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    full_dataset.to_csv(output_data, index=False)
            else:
                print("No data extracted.")
                return

    # 2. Training
    if full_dataset.empty:
        print("No data found to train on.")
        return
    
    trainer = ModelTrainer(models_dir=models_dir)
    
    # Train Qualifying
    q_data = full_dataset[full_dataset['SessionType'] == 'Qualifying']
    if not q_data.empty:
        trainer.train_qualifying(q_data)
        
    # Train Sprint
    s_data = full_dataset[full_dataset['SessionType'] == 'Sprint']
    if not s_data.empty:
        trainer.train_sprint(s_data)
        
    # Train Race
    r_data = full_dataset[full_dataset['SessionType'] == 'Race']
    if not r_data.empty:
        trainer.train_race(r_data)
        
    print(f"Training complete. Models saved to {models_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train F1 Prediction Models')
    parser.add_argument('--seasons', nargs='+', type=int, help='Seasons to fetch and train on (e.g. 2022 2023)')
    parser.add_argument('--models_dir', type=str, default='models', help='Output directory for models')
    parser.add_argument('--output_data', type=str, help='Path to save extracted features CSV')
    parser.add_argument('--input_data', nargs='+', type=str, help='Path(s) to existing features CSVs to train on (skips download)')
    
    args = parser.parse_args()
    
    if not args.seasons and not args.input_data:
        parser.error("You must provide either --seasons (to fetch data) or --input_data (to train on existing data).")
    
    train_on_real_data(args.seasons, args.models_dir, args.output_data, args.input_data)
