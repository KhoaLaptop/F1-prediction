import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import get_session_data

def test_load_monaco_2023():
    print("Testing data loading for Monaco 2023 Qualifying...")
    session = get_session_data(2023, 'Monaco', 'Q')
    
    if session:
        print(f"Successfully loaded session: {session.event['EventName']} - {session.name}")
        print(f"Number of laps: {len(session.laps)}")
        print(f"Weather data available: {not session.weather_data.empty}")
    else:
        print("Failed to load session.")

if __name__ == "__main__":
    test_load_monaco_2023()
