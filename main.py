import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline.predict import Predictor

def main():
    parser = argparse.ArgumentParser(description='F1 Performance Predictor')
    parser.add_argument('--driver', type=str, help='Driver name (required unless --realtime is set)')
    parser.add_argument('--gp', type=str, help='Grand Prix name (required unless --realtime is set)')
    parser.add_argument('--season', type=int, help='Season year (required unless --realtime is set)')
    parser.add_argument('--models_dir', type=str, default='models', help='Directory containing trained models')
    parser.add_argument('--realtime', action='store_true', help='Predict for the next upcoming session automatically')
    parser.add_argument('--weather', type=float, help='Override weather rain probability (0.0 to 1.0)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.realtime:
        if not args.driver or not args.gp or not args.season:
            parser.error("--driver, --gp, and --season are required unless --realtime is set")
            
    try:
        predictor = Predictor(models_dir=args.models_dir)
        
        if args.realtime:
            print("Running in Real-Time Mode...")
            # If no driver specified, predict for all
            driver = args.driver if args.driver else None
            predictions = predictor.predict_next_session(driver_name=driver, weather_override=args.weather)
            
            if isinstance(predictions, list):
                print("\nPredictions:")
                print(f"{'Pos':<4} {'Driver':<8} {'Quali':<6} {'Sprint':<10} {'Race Score'}")
                print("-" * 45)
                for p in predictions:
                    pos = p.get('Predicted_Position', '-')
                    drv = p.get('Driver', 'UNK')
                    quali = p.get('Qualifying_Position', 0)
                    if isinstance(quali, float):
                        quali = f"P{int(quali)}"
                    else:
                        quali = f"P{quali}"
                        
                    sprint = p.get('Sprint_Class', 'N/A')
                    score = p.get('Race_Score', 0)
                    
                    print(f"{pos:<4} {drv:<8} {quali:<6} {sprint:<10} {score:.4f}")
            elif predictions:
                print(f"\nPrediction for {driver}:")
                print(f"Qualifying Position: {predictions['Qualifying_Position']:.2f}")
                print(f"Sprint Classification: {predictions['Sprint_Class']}")
                print(f"Race Score (Rank): {predictions['Race_Score']:.4f}")
                if 'Predicted_Position' in predictions:
                    print(f"Predicted Finish: P{predictions['Predicted_Position']}")
            else:
                print("No predictions generated.")
            return
        else:
            predictions = predictor.predict_driver(args.driver, args.season, args.gp)
        
        if predictions:
            print("\nPredictions:")
            
            if isinstance(predictions, list):
                # Print Leaderboard
                print(f"{'Pos':<4} {'Driver':<8} {'Quali':<6} {'Sprint':<10} {'Race Score':<10}")
                print("-" * 45)
                for p in predictions:
                    pos = p.get('Predicted_Position', '-')
                    drv = p.get('Driver', 'UNK')
                    quali = p.get('Qualifying_Position', 0)
                    if isinstance(quali, float):
                        quali = f"P{int(quali)}"
                    else:
                        quali = f"P{quali}"
                        
                    sprint = p.get('Sprint_Class', 'N/A')
                    score = p.get('Race_Score', 0)
                    
                    print(f"{pos:<4} {drv:<8} {quali:<6} {sprint:<10} {score:.4f}")
            else:
                # Single driver
                print(f"Qualifying Position: {predictions['Qualifying_Position']:.2f}")
                print(f"Sprint Classification: {predictions['Sprint_Class']}")
                print(f"Race Score (Rank): {predictions['Race_Score']:.4f}")
                if 'Predicted_Position' in predictions:
                    print(f"Predicted Finish: P{predictions['Predicted_Position']}")
        else:
            print("Failed to generate predictions.")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure models are trained. Run 'python tests/test_models.py' to generate dummy models if needed.")

if __name__ == "__main__":
    main()
