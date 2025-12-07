import pandas as pd
import numpy as np

def calculate_race_pace(session, driver, min_laps=5):
    """
    Calculate race pace, tire degradation, top speed, and rain probability from a practice session.
    
    Args:
        session: FastF1 session object (usually FP2).
        driver: Driver abbreviation (e.g. 'VER').
        min_laps: Minimum consecutive laps to consider a "long run".
        
    Returns:
        tuple: (race_pace, tire_deg, top_speed, rain_prob)
            race_pace (float): Average lap time in seconds.
            tire_deg (float): Average seconds lost per lap (slope).
            top_speed (float): Maximum speed recorded in km/h.
            rain_prob (float): Percentage of session with rainfall (0.0 to 1.0).
    """
    if session is None:
        return None, None, None, None
        
    try:
        # 1. Rain Probability (Session Level)
        # Check weather data
        rain_prob = 0.0
        if hasattr(session, 'weather_data') and session.weather_data is not None:
            # Calculate % of samples where Rainfall is True
            rain_samples = session.weather_data['Rainfall'].sum()
            total_samples = len(session.weather_data)
            if total_samples > 0:
                rain_prob = float(rain_samples) / total_samples
        
        laps = session.laps.pick_drivers(driver)
        if laps.empty:
            return None, None, None, rain_prob
            
        # 2. Top Speed (Driver Level)
        # Use telemetry from fastest lap or all laps?
        # Let's use the max speed from the fastest lap to be efficient, 
        # or max speed from all "quick" laps.
        # Accessing telemetry for all laps is slow.
        # Let's try to get 'ST' (Speed Trap) column from laps if available.
        top_speed = None
        if 'ST' in laps.columns:
            top_speed = laps['ST'].max()
        
        # If ST is missing or NaN, try telemetry of fastest lap
        if pd.isna(top_speed):
            fastest = laps.pick_fastest()
            if fastest is not None:
                try:
                    tel = fastest.get_telemetry()
                    top_speed = tel['Speed'].max()
                except:
                    pass
        
        # Filter for valid laps only for Race Pace
        laps = laps.pick_quicklaps().pick_track_status('1') # Track status 1 = Green flag
        
        if laps.empty:
            return None, None, top_speed, rain_prob
            
        # Identify stints
        long_run_avgs = []
        long_run_slopes = []
        
        for stint_id in laps['Stint'].unique():
            stint_laps = laps[laps['Stint'] == stint_id]
            
            # Check length
            if len(stint_laps) >= min_laps:
                # Calculate average lap time for this stint
                times = stint_laps['LapTime'].dt.total_seconds().values
                
                # Remove outliers?
                # Simple mean for pace
                avg_time = np.mean(times)
                long_run_avgs.append(avg_time)
                
                # Calculate degradation (slope)
                x = np.arange(len(times))
                if len(x) > 1:
                    slope, _ = np.polyfit(x, times, 1)
                    long_run_slopes.append(slope)
                
        final_pace = None
        final_deg = None
        
        if long_run_avgs:
            final_pace = np.mean(long_run_avgs)
            final_deg = np.mean(long_run_slopes) if long_run_slopes else 0.0
            
        return final_pace, final_deg, top_speed, rain_prob
            
    except Exception as e:
        print(f"Error calculating race pace for {driver}: {e}")
        return None, None, None, None
