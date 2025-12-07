import pandas as pd

# Static lookup for overtake difficulty (0-10)
# This is a simplified example dictionary.
OVERTAKE_DIFFICULTY = {
    'Monaco': 10,
    'Singapore': 8,
    'Hungaroring': 7,
    'Silverstone': 3,
    'Spa-Francorchamps': 2,
    'Monza': 1,
    # Add more as needed or use default
}

def get_track_overtake_difficulty(circuit_name):
    """
    Get the historical overtake difficulty score for a track.
    """
    # Normalize name if needed
    for key in OVERTAKE_DIFFICULTY:
        if key in circuit_name:
            return OVERTAKE_DIFFICULTY[key]
            
    return 5 # Default average difficulty

def calculate_track_temp_avg(session):
    """
    Average track temperature during the session.
    """
    try:
        if session.weather_data is None or session.weather_data.empty:
            return 25.0 # Default fallback
        return session.weather_data['TrackTemp'].mean()
    except Exception:
        return 25.0

