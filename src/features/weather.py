"""
Weather API integration for F1 predictions.
Uses OpenWeatherMap API to fetch real weather forecasts.
"""

import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# F1 Circuit GPS Coordinates (lat, lon)
CIRCUIT_COORDINATES = {
    # 2024/2025 Calendar
    'Bahrain': (26.0325, 50.5106),
    'Saudi Arabia': (21.6319, 39.1044),
    'Jeddah': (21.6319, 39.1044),
    'Australia': (-37.8497, 144.9680),
    'Melbourne': (-37.8497, 144.9680),
    'Japan': (34.8431, 136.5407),
    'Suzuka': (34.8431, 136.5407),
    'China': (31.3389, 121.2197),
    'Shanghai': (31.3389, 121.2197),
    'Miami': (25.9581, -80.2389),
    'Imola': (44.3439, 11.7167),
    'Emilia Romagna': (44.3439, 11.7167),
    'Monaco': (43.7347, 7.4206),
    'Monte Carlo': (43.7347, 7.4206),
    'Canada': (45.5000, -73.5228),
    'Montreal': (45.5000, -73.5228),
    'Spain': (41.5700, 2.2611),
    'Barcelona': (41.5700, 2.2611),
    'Austria': (47.2197, 14.7647),
    'Spielberg': (47.2197, 14.7647),
    'United Kingdom': (52.0786, -1.0169),
    'Silverstone': (52.0786, -1.0169),
    'Hungary': (47.5789, 19.2486),
    'Budapest': (47.5789, 19.2486),
    'Belgium': (50.4372, 5.9714),
    'Spa': (50.4372, 5.9714),
    'Netherlands': (52.3888, 4.5409),
    'Zandvoort': (52.3888, 4.5409),
    'Italy': (45.6156, 9.2811),
    'Monza': (45.6156, 9.2811),
    'Azerbaijan': (40.3725, 49.8533),
    'Baku': (40.3725, 49.8533),
    'Singapore': (1.2914, 103.8644),
    'Marina Bay': (1.2914, 103.8644),
    'United States': (30.1328, -97.6411),
    'Austin': (30.1328, -97.6411),
    'Mexico': (19.4042, -99.0907),
    'Mexico City': (19.4042, -99.0907),
    'Brazil': (-23.7036, -46.6997),
    'São Paulo': (-23.7036, -46.6997),
    'Interlagos': (-23.7036, -46.6997),
    'Las Vegas': (36.1147, -115.1728),
    'Qatar': (25.4900, 51.4542),
    'Lusail': (25.4900, 51.4542),
    'Abu Dhabi': (24.4672, 54.6031),
    'Yas Marina': (24.4672, 54.6031),
}


def get_circuit_coordinates(circuit_name):
    """
    Get GPS coordinates for a circuit.
    Tries to match the circuit name to known circuits.
    """
    # Try direct match first
    if circuit_name in CIRCUIT_COORDINATES:
        return CIRCUIT_COORDINATES[circuit_name]
    
    # Try partial match
    for key, coords in CIRCUIT_COORDINATES.items():
        if key.lower() in circuit_name.lower() or circuit_name.lower() in key.lower():
            return coords
    
    # Default to a generic location (London) if not found
    print(f"Warning: Could not find coordinates for {circuit_name}. Using default.")
    return (51.5074, -0.1278)


def get_weather_forecast(lat, lon, target_datetime=None):
    """
    Fetch weather forecast from OpenWeatherMap API.
    
    Args:
        lat: Latitude
        lon: Longitude
        target_datetime: Optional datetime for the forecast. 
                        If None, returns current weather.
    
    Returns:
        dict: {'temperature': float, 'rain_probability': float (0-1), 'description': str}
    """
    api_key = os.getenv('OPENWEATHER_API_KEY')
    
    if not api_key:
        print("Warning: OPENWEATHER_API_KEY not set. Using default weather.")
        return {'temperature': 25.0, 'rain_probability': 0.0, 'description': 'Unknown'}
    
    try:
        # Use 5-day forecast API if target_datetime is in the future
        if target_datetime and target_datetime > datetime.now():
            url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if response.status_code != 200:
                print(f"API Error: {data.get('message', 'Unknown error')}")
                return {'temperature': 25.0, 'rain_probability': 0.0, 'description': 'Error'}
            
            # Find the closest forecast to target_datetime
            forecasts = data.get('list', [])
            closest_forecast = None
            min_diff = float('inf')
            
            for forecast in forecasts:
                forecast_dt = datetime.fromtimestamp(forecast['dt'])
                diff = abs((forecast_dt - target_datetime).total_seconds())
                if diff < min_diff:
                    min_diff = diff
                    closest_forecast = forecast
            
            if closest_forecast:
                temp = closest_forecast['main']['temp']
                # Rain probability is in 'pop' field (0-1)
                rain_prob = closest_forecast.get('pop', 0.0)
                description = closest_forecast['weather'][0]['description']
                return {'temperature': temp, 'rain_probability': rain_prob, 'description': description}
        
        # Use current weather API
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if response.status_code != 200:
            print(f"API Error: {data.get('message', 'Unknown error')}")
            return {'temperature': 25.0, 'rain_probability': 0.0, 'description': 'Error'}
        
        temp = data['main']['temp']
        # Current weather doesn't have 'pop', infer from weather condition
        weather_main = data['weather'][0]['main'].lower()
        rain_prob = 1.0 if 'rain' in weather_main else 0.0
        description = data['weather'][0]['description']
        
        return {'temperature': temp, 'rain_probability': rain_prob, 'description': description}
        
    except Exception as e:
        print(f"Error fetching weather: {e}")
        return {'temperature': 25.0, 'rain_probability': 0.0, 'description': 'Error'}


def get_race_weather(circuit_name, race_datetime=None):
    """
    Get weather forecast for a race.
    
    Args:
        circuit_name: Name of the circuit (e.g., 'Monaco', 'Silverstone')
        race_datetime: Optional datetime of the race. If None, uses current weather.
    
    Returns:
        dict: {'temperature': float, 'rain_probability': float, 'description': str}
    """
    lat, lon = get_circuit_coordinates(circuit_name)
    return get_weather_forecast(lat, lon, race_datetime)


if __name__ == '__main__':
    # Test the module
    print("Testing Weather API...")
    
    # Test Abu Dhabi
    result = get_race_weather('Abu Dhabi')
    print(f"Abu Dhabi current weather: {result}")
    
    # Test Monaco with future date
    future = datetime.now() + timedelta(days=2)
    result = get_race_weather('Monaco', future)
    print(f"Monaco forecast (in 2 days): {result}")
