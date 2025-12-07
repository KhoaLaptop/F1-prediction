import pandas as pd

def calculate_constructor_standing(standings_data, team_name, current_round):
    """
    Get current rank of the team in the constructors championship.
    """
    # This requires standings data which might be separate from session data.
    # For now, we'll assume we pass a dataframe of standings history.
    if standings_data is None or standings_data.empty:
        return np.nan
        
    # Filter for current round (or previous round to be predictive)
    # Usually we use standing AFTER previous round.
    prev_round = current_round - 1
    if prev_round < 1:
        return 0 # Start of season
        
    team_standing = standings_data[(standings_data['RoundNumber'] == prev_round) & (standings_data['TeamName'] == team_name)]
    
    if team_standing.empty:
        return np.nan
        
    return team_standing['Position'].iloc[0]

def calculate_reliability_score(constructor_results, current_round, window=10):
    """
    Inverse frequency of mechanical failures over last 10 races.
    """
    # Similar to driver DNF but aggregated for team.
    past_results = constructor_results[constructor_results['RoundNumber'] < current_round].sort_values('RoundNumber')
    
    if past_results.empty:
        return 1.0 # Perfect reliability start
        
    recent = past_results.tail(window * 2) # 2 cars per team
    
    failures = 0
    for _, row in recent.iterrows():
        status = str(row['Status']).lower()
        if 'engine' in status or 'gearbox' in status or 'hydraulics' in status or 'mechanical' in status:
            failures += 1
            
    total_entries = len(recent)
    if total_entries == 0:
        return 1.0
        
    return 1.0 - (failures / total_entries)
