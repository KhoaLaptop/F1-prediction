import fastf1
import os
import pandas as pd
from datetime import datetime
import pytz
import time

# Enable caching
CACHE_DIR = 'fastf1_cache'
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

fastf1.Cache.enable_cache(CACHE_DIR)

def load_season_schedule(year):
    """
    Load the event schedule for a specific season.
    """
    schedule = fastf1.get_event_schedule(year)
    return schedule

def get_session_data(year, grand_prix, session_type, load_telemetry=True, load_weather=True):
    """
    Load data for a specific session.
    
    Args:
        year (int): Season year.
        grand_prix (str or int): Name or round number of the Grand Prix.
        session_type (str): 'Q', 'S', 'R', 'FP1', 'FP2', 'FP3'.
        load_telemetry (bool): Whether to load telemetry data.
        load_weather (bool): Whether to load weather data.
        
    Returns:
        fastf1.core.Session: The loaded session object.
    """
    try:
        session = fastf1.get_session(year, grand_prix, session_type)
        session.load(telemetry=load_telemetry, weather=load_weather)
        return session
    except Exception as e:
        print(f"Error loading session {session_type} for {grand_prix} {year}: {e}")
        return None

def load_all_season_sessions(year, session_types=['Q', 'R']):
    """
    Load all sessions of specified types for a given season.
    """
    schedule = load_season_schedule(year)
    sessions = []
    
    # Filter for completed events (this is a simplification, might need more robust check)
    # fastf1 schedule has 'EventDate'
    
    for i, row in schedule.iterrows():
        # Skip testing
        if 'Test' in row['EventName']:
            continue
            
        round_number = row['RoundNumber']
        event_name = row['EventName']
        
        print(f"Loading data for Round {round_number}: {event_name}")
        
        event_sessions = {}
        for st in session_types:
            # Check if session exists (e.g. Sprint might not exist)
            # FastF1 handles this gracefully usually, but we can check schedule columns
            # Schedule has columns like 'Session1', 'Session2', etc. mapping to types
            
            # Simple approach: try to load
            session = get_session_data(year, round_number, st)
            if session:
                event_sessions[st] = session
        
        if event_sessions:
            sessions.append({
                'round': round_number,
                'event_name': event_name,
                'sessions': event_sessions
            })
        
        # Add a delay to avoid rate limiting
        print("Sleeping for 2 seconds to respect API limits...")
        time.sleep(2)
            
    return sessions

def get_next_event(year=None):
    """
    Find the next upcoming or currently active event.
    """
    if year is None:
        year = datetime.now().year
        
    schedule = load_season_schedule(year)
    
    # Get current time (naive)
    now = datetime.now()
    
    # FastF1 schedule 'Session5Date' is usually the Race date.
    # We want the first event where the Race has not finished yet.
    # Or more granularly, the next session.
    
    # Let's find the first round where the Race date is in the future or today.
    # Note: FastF1 dates might be naive or tz-aware. Usually they are naive UTC in the dataframe?
    # Actually FastF1 returns them as datetime objects. Let's assume they are UTC-ish or we compare naive.
    
    # Safe comparison
    upcoming = fastf1.get_events_remaining(dt=now)
    
    if upcoming.empty:
        return None
        
    return upcoming.iloc[0]

