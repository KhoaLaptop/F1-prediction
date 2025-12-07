import fastf1
from datetime import datetime

def check_latest_quali():
    # Enable cache
    fastf1.Cache.enable_cache('fastf1_cache')
    
    # Get the session
    # We know from logs it's Abu Dhabi 2025
    session = fastf1.get_session(2025, 'Abu Dhabi', 'Q')
    session.load(telemetry=False, weather=False, messages=False)
    
    print(f"Session: {session.event['EventName']} {session.name}")
    
    if hasattr(session, 'results') and not session.results.empty:
        # Get Max's result
        ver = session.results.loc[session.results['Abbreviation'] == 'VER']
        if not ver.empty:
            pos = ver['Position'].iloc[0]
            print(f"Max Verstappen (VER) Position: {pos}")
        else:
            print("VER not found in results.")
            
        # Print top 3
        print("\nTop 3:")
        print(session.results[['Abbreviation', 'Position', 'TeamName']].head(3))
    else:
        print("No results found.")

if __name__ == "__main__":
    check_latest_quali()
