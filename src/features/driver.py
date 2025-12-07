import pandas as pd
import numpy as np

def calculate_driver_season_avg_position(driver_results, current_round, window=5):
    """
    Calculate average finishing position over the last N races.
    """
    # Filter for races before current round in the same season
    past_results = driver_results[driver_results['RoundNumber'] < current_round].sort_values('RoundNumber')
    
    if past_results.empty:
        return np.nan
        
    recent_results = past_results.tail(window)
    # Use 'Position' column, handle NaNs or DNFs if necessary (usually Position is numeric)
    # FastF1 Position is float.
    return recent_results['Position'].mean()

def calculate_driver_dnf_rate(driver_results, current_round):
    """
    Calculate percentage of races not finished in current season.
    """
    past_results = driver_results[driver_results['RoundNumber'] < current_round]
    
    if past_results.empty:
        return 0.0
        
    # Check 'Status' column. 'Finished' or '+1 Lap' etc are good.
    # FastF1 has 'ClassifiedPosition' which is a string number or 'R', 'D', 'N', 'W'
    # Or check 'Status' string.
    
    # Simple heuristic: if Position is NaN or Status is not 'Finished'/'... Lap'
    # Actually FastF1 'Position' is NaN for DNF usually?
    # Let's use 'Status'. Common DNF statuses: 'Collision', 'Engine', 'Gearbox', etc.
    # 'Finished' is good.
    
    total_races = len(past_results)
    # Count non-finishers.
    # A robust way is checking if 'ClassifiedPosition' is numeric.
    
    dnfs = 0
    for _, row in past_results.iterrows():
        status = str(row['Status'])
        if status not in ['Finished'] and '+' not in status:
             # This is a simplification.
             dnfs += 1
             
    return dnfs / total_races

def calculate_qualifying_delta_to_teammate(session, driver, teammate):
    """
    Time difference between driver and teammate in Q3 (or last participated session).
    """
    if not teammate:
        return 0.0
        
    # Get laps for driver and teammate
    d_laps = session.laps.pick_drivers(driver)
    t_laps = session.laps.pick_drivers(teammate)
    
    if d_laps.empty or t_laps.empty:
        return 0.0
        
    d_fastest = d_laps.pick_fastest()
    t_fastest = t_laps.pick_fastest()
    
    if d_fastest is None or t_fastest is None:
        return 0.0
        
    if pd.isnull(d_fastest['LapTime']) or pd.isnull(t_fastest['LapTime']):
        return 0.0
        
    delta = d_fastest['LapTime'] - t_fastest['LapTime']
    return delta.total_seconds()
