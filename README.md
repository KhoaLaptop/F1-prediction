# F1 Race Prediction System 🏎️

An AI-powered Formula 1 race prediction system using XGBoost machine learning models. Predicts qualifying positions, sprint classifications, and race finishing positions based on historical data, practice session analysis, and real-time weather forecasts.

## Features

- **Real-time Predictions**: Automatically identifies the next upcoming race and generates predictions
- **Weather Integration**: Fetches live weather forecasts from OpenWeatherMap API
- **Practice Analysis**: Extracts race pace, tire degradation, and top speed from practice sessions
- **Historical Stats**: Uses driver and constructor statistics for accurate predictions
- **Multiple Models**: Separate XGBoost models for Qualifying, Sprint, and Race sessions

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/KhoaLaptop/F1-prediction.git
cd F1-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Weather API (Optional)
Get a free API key from [OpenWeatherMap](https://openweathermap.org/api) and create a `.env` file:
```bash
echo "OPENWEATHER_API_KEY=your_api_key_here" > .env
```

### 4. Train Models (First Time Only)
```bash
python train.py --seasons 2022 2023 2024 2025
```

### 5. Run Predictions
```bash
python main.py --realtime
```

## Usage Examples

### Predict Next Race (Auto Weather)
```bash
python main.py --realtime
```

### Predict with Weather Override
```bash
# Dry race
python main.py --realtime --weather 0.0

# Wet race (100% rain)
python main.py --realtime --weather 1.0

# 50% rain chance
python main.py --realtime --weather 0.5
```

### Predict Specific Driver
```bash
python main.py --realtime --driver VER
```

## Sample Output
```
Next event: Abu Dhabi Grand Prix (2025)
Qualifying appears to be finished. Predicting Race.
Fetching live weather forecast...
  Weather: clear sky, Temp: 23.1°C, Rain: 0%

Predictions:
Pos  Driver   Quali  Sprint     Race Score
---------------------------------------------
1    VER      P1     N/A        3.5607
2    NOR      P2     N/A        2.3424
3    BOR      P7     N/A        1.4970
4    RUS      P4     N/A        1.3159
5    HAD      P9     N/A        1.1569
...
```

## Model Features

| Feature | Description |
|---------|-------------|
| `TrackTemp` | Track temperature from session weather |
| `OvertakeDifficulty` | Circuit-specific difficulty rating (Monaco=9, Monza=3) |
| `DriverAvgPos` | Historical average finishing position |
| `DriverDNFRate` | Historical DNF rate |
| `ReliabilityScore` | Team reliability rating |
| `QualiDeltaTeammate` | Qualifying gap to teammate |
| `GridPosition` | Starting position (from qualifying) |
| `RacePace` | Long-run pace from practice sessions |
| `TireDegradation` | Tire wear rate from practice (sec/lap) |
| `TopSpeed` | Maximum speed from practice speed traps |
| `RainProbability` | Weather forecast rain chance |

## Project Structure
```
F1-prediction/
├── main.py              # Entry point
├── train.py             # Model training script
├── src/
│   ├── data/            # Data loading utilities
│   ├── features/        # Feature extraction
│   │   ├── practice.py  # Race pace, tire deg, top speed
│   │   ├── weather.py   # OpenWeatherMap integration
│   │   └── ...
│   ├── models/          # XGBoost model definitions
│   └── pipeline/        # Prediction pipeline
├── models/              # Trained model files
└── requirements.txt     # Dependencies
```

## Requirements

- Python 3.10+
- FastF1 (F1 data API)
- XGBoost
- pandas, numpy
- python-dotenv (for API keys)
- requests (for weather API)

## License

MIT License
