
import os
import pandas as pd
import numpy as np
from src.models.qualifying import QualifyingModel
from src.models.sprint import SprintModel
from src.models.race import RaceModel
from src.features.processor import extract_features_for_session
from src.data.loader import get_session_data, get_next_event, load_season_schedule
from src.features import driver, constructor, track, practice, weather
from datetime import datetime
import pytz

class Predictor:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.quali_model = QualifyingModel.load(os.path.join(models_dir, 'qualifying_model.pkl'))
        self.sprint_model = SprintModel.load(os.path.join(models_dir, 'sprint_model.pkl'))
        self.race_model = RaceModel.load(os.path.join(models_dir, 'race_model.pkl'))
        
        # Load historical data
        self.history_df = pd.DataFrame()
        # Try to load available feature files
        for year in range(2022, 2026):
            path = f"features_{year}.csv"
            if os.path.exists(path):
                print(f"Loading history from {path}...")
                df = pd.read_csv(path)
                self.history_df = pd.concat([self.history_df, df])
        
        if self.history_df.empty:
            print("Warning: No historical data found (features_YYYY.csv). Predictions will use defaults.")
        else:
            print(f"Loaded history for {len(self.history_df)} races.")
            print(f"Drivers in history: {self.history_df['Driver'].unique()[:10]}...")
            
    def calculate_driver_stats(self, driver_name):
        """
        Calculate driver stats from loaded history.
        """
        if self.history_df.empty:
            return 5.0, 0.1, 0.95 # Defaults
            
        # Map driver abbreviation (e.g. VER) to number (e.g. 1 or 33)
        # We can try to infer this from the session object if we had it, 
        # or use a static map, or check if 'driver_name' is already a number.
        
        # Simple static map for top drivers (expand as needed or fetch dynamically)
        # Actually, let's try to find the driver number in the current session data if available?
        # But calculate_driver_stats is called before we might have session data fully loaded or if we want to predict without it.
        
        # Better approach: The history dataframe has 'Driver' column as numbers (int or str).
        # We need to find which number corresponds to 'VER'.
        # We don't have that mapping in the history DF itself (it just has 'Driver' column).
        # Wait, we might have 'TeamName'?
        
        # Let's use a hardcoded map for now for the main drivers to fix the immediate issue.
        # In a real app, we'd load a driver list.
        
        driver_map = {
            'VER': '1', 'PER': '11',
            'LEC': '16', 'SAI': '55',
            'NOR': '4', 'PIA': '81',
            'HAM': '44', 'RUS': '63',
            'ALO': '14', 'STR': '18',
            'TSU': '22', 'RIC': '3',
            'ALB': '23', 'SAR': '2',
            'ZHO': '24', 'BOT': '77',
            'MAG': '20', 'HUL': '27',
            'GAS': '10', 'OCO': '31',
            'COL': '43', 'BEA': '87', 'LAW': '30', 'DOO': '12' # Rookies/Subs
        }
        
        driver_id = driver_map.get(driver_name, driver_name) # Fallback to using input as ID
        
        # Ensure ID is string for comparison if DF has strings, or int if DF has ints.
        # The debug output showed [16 1 55 ...], likely integers or strings.
        # Let's try both or convert column to string.
        
        # Convert history driver column to string for consistent comparison
        self.history_df['Driver'] = self.history_df['Driver'].astype(str)
        driver_id = str(driver_id)
        
        d_hist = self.history_df[self.history_df['Driver'] == driver_id]
        
        if d_hist.empty:
            # Try mapping? Or just return defaults
            return 5.0, 0.1, 0.95
            
        # 1. Avg Position (last 5 races)
        # Sort by Year, RoundNumber if available, else just take tail
        # We don't have Year/Round in features CSV explicitly unless we added it.
        # Wait, features CSV has 'DriverAvgPos' which IS the rolling avg at that point in time.
        # So we can just take the *latest* value from their history?
        # NO, 'DriverAvgPos' in the CSV is the feature used for THAT race.
        # If we want to predict for a NEW race, we need the avg pos *after* the last known race.
        # But 'DriverAvgPos' is "avg pos of previous 5 races".
        # So if we take the 'DriverAvgPos' from the very last race in history, that is avg of races [N-5, N-1].
        # We actually want avg of [N-4, N].
        # So we should re-calculate it from raw positions if possible.
        # But our features CSV doesn't store raw results row-by-row in a simple way, it stores FEATURES.
        # Ah, wait. The features CSV *is* the training data.
        # It contains 'DriverAvgPos'.
        # It does NOT contain the raw result of that race (Position) as a feature, but as a target 'Position'.
        
        # Let's look at the columns in features CSV.
        # It has 'Position' (target).
        
        # So we can take the last 5 'Position' values from history and average them.
        # Assuming rows are somewhat chronological.
        
        # Filter for Race sessions only to calculate avg race pos
        r_hist = d_hist[d_hist['SessionType'] == 'Race']
        
        if r_hist.empty:
             return 5.0, 0.1, 0.95
             
        # Take last 5 valid positions
        # We need to ensure we don't include NaNs (DNFs) in position calculation if we want pure finishing pos,
        # or we handle them.
        # In feature extraction: "avg_pos = recent_results['Position'].mean()" where recent_results excludes DNFs usually.
        
        # Column is 'TargetPosition' in features CSV
        valid_pos = r_hist['TargetPosition'].dropna()
        if valid_pos.empty:
             return 5.0, 0.1, 0.95
             
        avg_pos = valid_pos.tail(5).mean()
        
        # 2. DNF Rate
        # DNF is when Status != 'Finished' and != '+1 Lap' etc.
        # But we don't have 'Status' column in features CSV?
        # Let's check what columns we saved.
        # train.py: full_dataset.to_csv.
        # full_dataset is concat of feats.
        # feats comes from extract_features_for_session.
        # extract_features_for_session returns a DataFrame with features AND target 'Position'.
        # Does it return 'Status'?
        # Let's check src/features/processor.py.
        # It returns: 'Driver', 'Team', 'RoundNumber', 'Position', 'Status', plus features.
        
        # So we DO have 'Status'.
        
        total_races = len(r_hist)
        # Simple DNF check: Position is NaN usually implies DNF in our extraction?
        # Or check 'Status'.
        # Let's use 'Status' if available.
        
        dnf_count = 0
        # Check if 'Status' column exists, otherwise infer from TargetPosition
        # Actually, TargetPosition might be NaN for DNFs?
        
        dnf_count = r_hist['TargetPosition'].isna().sum()
        
        dnf_rate = dnf_count / total_races if total_races > 0 else 0.0
        
        # 3. Reliability Score
        # Similar to DNF rate but maybe team based.
        # Let's just use (1 - dnf_rate) for simplicity or look at 'ReliabilityScore' column from last race.
        # The 'ReliabilityScore' feature is usually Constructor reliability.
        # Let's take the last known 'ReliabilityScore' for this driver (proxy for team).
        
        if 'ReliabilityScore' in d_hist.columns:
            rel_score = d_hist['ReliabilityScore'].iloc[-1]
        else:
            rel_score = 0.95
            
        return avg_pos, dnf_rate, rel_score
        
    def predict_driver(self, driver_name, season, grand_prix, actual_quali_pos=None, practice_session=None, weather_override=None):
        """
        Generate predictions for a driver in a specific GP.
        """
        print(f"Generating predictions for {driver_name} at {grand_prix} {season}...")
        
        # 1. Load Session Data (We need the session to get track info etc)
        # For prediction, we might not have the session data yet if it's in the future.
        # But for this demo, we assume we are predicting for a session that exists (or we mock it).
        # Let's try to load the session.
        session = get_session_data(season, grand_prix, 'Q') # Load Quali as base
        
        # Mock features if session fails
        from src.features import track
        
        if session:
            try:
                track_temp = track.calculate_track_temp_avg(session)
            except:
                print("Warning: Could not load session weather data. Using default.")
                track_temp = 30.0
        else:
            print("Warning: Could not load session weather data. Using default.")
            track_temp = 30.0
            
        overtake_diff = track.get_track_overtake_difficulty(grand_prix)
        
        # Calculate real stats from history
        driver_avg_pos, driver_dnf_rate, rel_score = self.calculate_driver_stats(driver_name)
        quali_delta = 0.0 # Still hard to calc without teammate info for *next* race easily
        
        print(f"  Using historical stats for {driver_name}: AvgPos={driver_avg_pos:.2f}, DNF={driver_dnf_rate:.2f}, Rel={rel_score:.2f}")

        # 2. Extract Features
        # Load Practice session for Race Pace
        # We need to know if it's Sprint or Conventional.
        # Try FP2 then FP1.
        # The practice_session is now passed in from predict_race or predict_next_session
            
        # Extract features
        # We need to manually construct the feature dict or use processor?
        # The processor uses session.results which we don't have for the upcoming race (except entry list).
        # But predict_driver is usually called BEFORE the race.
        # We can reuse extract_features_for_session if we have a dummy session object with entry list?
        # Or we just calculate manually as before.
        
        # Race Pace
        race_pace = None
        tire_deg = None
        top_speed = None
        rain_prob = None
        
        if practice_session:
             race_pace, tire_deg, top_speed, rain_prob = practice.calculate_race_pace(practice_session, driver_name)
             
        # Handle missing race pace
        if race_pace is None:
            # Fallback? Maybe 0 or NaN.
            # If we use 0, it might be interpreted as super fast.
            # Let's use NaN and hope XGBoost handles it (it does).
            race_pace = np.nan
        if tire_deg is None:
            tire_deg = np.nan
        if top_speed is None:
            top_speed = np.nan
            
        # Weather Override
        if weather_override is not None:
            rain_prob = weather_override
        elif rain_prob is None:
            rain_prob = 0.0 # Default to 0 if unknown

        # Features for Qualifying Model (no GridPos, no RacePace)
        quali_features = pd.DataFrame([{
            'TrackTemp': track_temp,
            'OvertakeDifficulty': overtake_diff,
            'DriverAvgPos': driver_avg_pos,
            'DriverDNFRate': driver_dnf_rate,
            'QualiDeltaTeammate': quali_delta,
            'ReliabilityScore': rel_score
        }])

        # 3. Predict Qualifying
        q_pred = self.quali_model.predict(quali_features)[0]
        
        # Use actual qualifying position if available
        final_quali_pos = actual_quali_pos if actual_quali_pos is not None else q_pred
        grid_pos = final_quali_pos
        
        # Features for Race Model (includes GridPos and RacePace)
        race_features = pd.DataFrame([{
            'TrackTemp': track_temp,
            'OvertakeDifficulty': overtake_diff,
            'DriverAvgPos': driver_avg_pos,
            'DriverDNFRate': driver_dnf_rate,
            'ReliabilityScore': rel_score,
            'QualiDeltaTeammate': quali_delta,
            'GridPosition': grid_pos,
            'RacePace': race_pace,
            'TireDegradation': tire_deg,
            'TopSpeed': top_speed,
            'RainProbability': rain_prob
        }])

        # Check if Sprint weekend
        # FastF1 event object has 'EventFormat'
        is_sprint = False
        if session and hasattr(session, 'event'):
             # 'sprint' or 'sprint_shootout' or 'sprint_qualifying' implies sprint
             # Actually FastF1 EventFormat is 'sprint' or 'conventional' usually.
             if getattr(session.event, 'EventFormat', '') == 'sprint':
                 is_sprint = True
        
        # Sprint Model uses same features as Quali? Or Race?
        # SprintModel features: TrackTemp, OvertakeDifficulty, DriverAvgPos, DriverDNFRate, ReliabilityScore, QualiDeltaTeammate.
        # It does NOT use GridPosition or RacePace in current implementation (check sprint.py).
        # Let's check sprint.py content if needed. Assuming it uses basic features.
        
        if is_sprint:
            s_pred = self.sprint_model.predict(quali_features)[0]
        else:
            s_pred = "N/A"
            
        r_pred = self.race_model.predict(race_features)[0]
        
        return {
            'Qualifying_Position': final_quali_pos,
            'Sprint_Class': s_pred,
            'Race_Score': r_pred,
            'Grid_Position': grid_pos,
            'Race_Pace': race_pace,
            'Tire_Degradation': tire_deg,
            'Top_Speed': top_speed,
            'Rain_Probability': rain_prob
        }

    def predict_race(self, year, event_name, driver_list=None, weather_override=None):
        """
        Predict the full race classification.
        """
        print(f"Predicting race results for {event_name} {year}...")
        
        # Load session to get drivers if not provided
        session_q = get_session_data(year, event_name, 'Q', load_telemetry=False, load_weather=False)
        
        if not driver_list:
            if session_q and hasattr(session_q, 'results') and not session_q.results.empty:
                driver_list = session_q.results['Abbreviation'].tolist()
            else:
                # Fallback list
                driver_list = ['VER', 'PER', 'LEC', 'SAI', 'NOR', 'PIA', 'HAM', 'RUS', 'ALO', 'STR', 
                               'TSU', 'RIC', 'ALB', 'SAR', 'ZHO', 'BOT', 'MAG', 'HUL', 'GAS', 'OCO']
        
        # Load practice session for race pace calculation
        practice_session = get_session_data(year, event_name, 'FP2', load_telemetry=True) # FP2 is usually best for long runs
        if not practice_session:
            practice_session = get_session_data(year, event_name, 'FP1', load_telemetry=True) # Fallback to FP1
                               
        predictions = []
        for driver in driver_list:
            # Get actual quali pos if available
            actual_pos = None
            if session_q and hasattr(session_q, 'results') and not session_q.results.empty:
                 try:
                     res = session_q.results.loc[session_q.results['Abbreviation'] == driver]
                     if not res.empty:
                         actual_pos = res.iloc[0]['Position']
                 except:
                     pass
            
            pred = self.predict_driver(driver, year, event_name, actual_quali_pos=actual_pos, practice_session=practice_session, weather_override=weather_override)
            pred['Driver'] = driver
            predictions.append(pred)
            
        # Sort by Race Score (descending because higher relevance = better position in XGBRanker usually? 
        # Wait, in race.py we did: y_relevance = 21 - position. So Higher Score = Better Position (Lower Rank).
        # So we sort Descending.
        predictions.sort(key=lambda x: x['Race_Score'], reverse=True)
        
        # Assign ranks
        for i, p in enumerate(predictions):
            p['Predicted_Position'] = i + 1
            
        return predictions

    def predict_next_session(self, driver_name=None, weather_override=None):
        """
        Automatically identify the next session and predict for it.
        If driver_name is None, returns predictions for all drivers.
        """
        print("Identifying next event...")
        next_event = get_next_event()
        
        if next_event is None:
            print("No upcoming events found.")
            return None
            
        event_name = next_event['EventName']
        year = next_event['EventDate'].year
        print(f"Next event: {event_name} ({year})")
        
        try:
            # Check if Quali is done
            session_q = get_session_data(year, event_name, 'Q', load_telemetry=False, load_weather=False)
            
            # Get driver list for prediction
            driver_list = None
            if session_q and hasattr(session_q, 'results') and not session_q.results.empty:
                driver_list = session_q.results['Abbreviation'].tolist()
            else:
                # Fallback list if no session data yet
                driver_list = ['VER', 'PER', 'LEC', 'SAI', 'NOR', 'PIA', 'HAM', 'RUS', 'ALO', 'STR', 
                               'TSU', 'RIC', 'ALB', 'SAR', 'ZHO', 'BOT', 'MAG', 'HUL', 'GAS', 'OCO']
            
            if session_q and hasattr(session_q, 'results') and session_q.results is not None and not session_q.results.empty:
                print("Qualifying appears to be finished. Predicting Race.")
                
                # Fetch weather from API if no override provided
                if weather_override is None:
                    print("Fetching live weather forecast...")
                    circuit_name = next_event.get('Location', event_name)
                    race_datetime = next_event.get('Session5Date', None)  # Race is usually Session5
                    if race_datetime:
                        try:
                            race_datetime = race_datetime.to_pydatetime()
                        except:
                            pass
                    weather_data = weather.get_race_weather(circuit_name, race_datetime)
                    weather_override = weather_data['rain_probability']
                    print(f"  Weather: {weather_data['description']}, Temp: {weather_data['temperature']:.1f}°C, Rain: {weather_override*100:.0f}%")
                
                # Predict for ALL drivers to get ranking
                all_preds = self.predict_race(year, event_name, driver_list=driver_list, weather_override=weather_override)
                
                if driver_name:
                    # Find the requested driver
                    driver_pred = next((p for p in all_preds if p['Driver'] == driver_name), None)
                    
                    if driver_pred:
                        print(f"\nPredicted Finish for {driver_name}: P{driver_pred['Predicted_Position']}")
                        return driver_pred
                    else:
                        print(f"Driver {driver_name} not found in predictions.")
                        return None
                else:
                    # Return all predictions
                    return all_preds
            else:
                print("Qualifying not yet finished. Predicting Qualifying.")
                if driver_name:
                    return self.predict_driver(driver_name, year, event_name)
                else:
                    # Predict for default list if no driver specified for Quali
                    # Or implement predict_qualifying_grid
                    print("Predicting Qualifying for all drivers...")
                    # Reuse predict_race logic but for Quali? 
                    # predict_race calls predict_driver which does Quali prediction too.
                    # So we can just call predict_race (it sorts by Race Score but we can ignore that for Quali)
                    return self.predict_race(year, event_name)
                
        except Exception as e:
            print(f"Error checking session status: {e}")
            print("Defaulting to prediction for upcoming event...")
            if driver_name:
                return self.predict_driver(driver_name, year, event_name)
            else:
                 return self.predict_race(year, event_name)

