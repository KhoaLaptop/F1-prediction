import pandas as pd
from src.features import driver, constructor, track, practice

def extract_features_for_session(session, driver_history, constructor_history, practice_session=None):
    """
    Extract all features for all drivers in a given session.
    
    Args:
        session: FastF1 session object.
        driver_history: DataFrame containing historical results for drivers.
        constructor_history: DataFrame containing historical results for constructors.
        practice_session: Optional FastF1 session object (FP2) for race pace.
        
    Returns:
        DataFrame: Features for training/inference.
    """
    features = []
    
    round_num = session.event['RoundNumber']
    circuit_name = session.event['EventName'] # Or Location
    
    track_temp = track.calculate_track_temp_avg(session)
    overtake_diff = track.get_track_overtake_difficulty(circuit_name)
    
    # Iterate over drivers in the session
    # session.results contains the classification
    try:
        results = session.results
    except:
        # If results not available (e.g. future race?), we might need entry list
        # For training, we assume results exist.
        return pd.DataFrame()
        
    for driver_code, row in results.iterrows():
        driver_name = row['FullName'] # or Abbreviation
        driver_abbr = row['Abbreviation']
        team_name = row['TeamName']
        
        # Driver Features
        # Filter history for this driver
        # driver_history uses Abbreviation as 'Driver' column
        d_hist = driver_history[driver_history['Driver'] == driver_abbr]
        avg_pos = driver.calculate_driver_season_avg_position(d_hist, round_num)
        dnf_rate = driver.calculate_driver_dnf_rate(d_hist, round_num)
        
        # Teammate delta (requires identifying teammate)
        # Simple logic: find other driver with same TeamName
        teammate_row = results[(results['TeamName'] == team_name) & (results.index != driver_code)]
        teammate_code = teammate_row.index[0] if not teammate_row.empty else None
        
        delta_teammate = driver.calculate_qualifying_delta_to_teammate(session, driver_code, teammate_code)
        
        # Race Pace, Tire Deg, Top Speed, Rain Prob
        race_pace = None
        tire_deg = None
        top_speed = None
        rain_prob = None
        
        if practice_session:
            race_pace, tire_deg, top_speed, rain_prob = practice.calculate_race_pace(practice_session, driver_abbr)
            
        # Constructor Features
        c_hist = constructor_history[constructor_history['TeamName'] == team_name]
        rel_score = constructor.calculate_reliability_score(c_hist, round_num)
        
        # Combine
        feat = {
            'Driver': driver_code,
            'RoundNumber': round_num,
            'Circuit': circuit_name,
            'TrackTemp': track_temp,
            'OvertakeDifficulty': overtake_diff,
            'DriverAvgPos': avg_pos,
            'DriverDNFRate': dnf_rate,
            'QualiDeltaTeammate': delta_teammate,
            'ReliabilityScore': rel_score,
            'GridPosition': row.get('GridPosition', 0), # Use .get for safety
            'RacePace': race_pace,
            'TireDegradation': tire_deg,
            'TopSpeed': top_speed,
            'RainProbability': rain_prob,
            'TargetPosition': row['Position'] # Target variable
        }
        features.append(feat)
        
    return pd.DataFrame(features)
