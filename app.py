from flask import Flask, request, jsonify
import joblib
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Load saved model, scaler, and feature names
gb_model = joblib.load('eurovision_model.pkl')
scaler = joblib.load('scaler.pkl')

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

print("Model, scaler, and feature names loaded successfully!")

# ============================================================
# HELPER FUNCTION: Classify prediction
# ============================================================

def classify_score(predicted_score):
    """Classify the predicted score into Eurovision tiers"""
    if predicted_score < 50:
        return "Flop"
    elif predicted_score < 150:
        return "Poor"
    elif predicted_score < 250:
        return "Average"
    elif predicted_score < 350:
        return "Good"
    elif predicted_score < 450:
        return "Excellent"
    else:
        return "Winner"

# ============================================================
# HEALTH CHECK ENDPOINT
# ============================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'Model is running',
        'timestamp': datetime.now().isoformat()
    }), 200

# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict Eurovision score based on audio features
    
    Expected JSON input:
    {
        "danceability": 0.7,
        "energy": 0.8,
        "valence": 0.6,
        "tempo": 120,
        "acousticness": 0.1
    }
    
    Returns:
    {
        "success": true,
        "predicted_score": 285.5,
        "confidence": "68%",
        "classification": "Good",
        "mae_margin": "±41.95",
        "prediction_range": "243.55-327.45",
        "interpretation": "Song expected to perform well",
        "timestamp": "2024-06-29T10:30:45.123456"
    }
    """
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate that data is provided
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided',
                'expected_format': {
                    'danceability': 'float (0-1)',
                    'energy': 'float (0-1)',
                    'valence': 'float (0-1)',
                    'tempo': 'float (BPM)',
                    'acousticness': 'float (0-1)'
                }
            }), 400
        
        # Extract features
        features = []
        missing_features = []
        
        for feature in feature_names:
            if feature in data:
                features.append(data[feature])
            else:
                missing_features.append(feature)
        
        # Check if all features are provided
        if missing_features:
            return jsonify({
                'success': False,
                'error': f'Missing features: {missing_features}',
                'required_features': feature_names
            }), 400
        
        # Validate feature values
        errors = []
        if not (0 <= data['danceability'] <= 1):
            errors.append('danceability must be between 0 and 1')
        if not (0 <= data['energy'] <= 1):
            errors.append('energy must be between 0 and 1')
        if not (0 <= data['valence'] <= 1):
            errors.append('valence must be between 0 and 1')
        if not (0 <= data['tempo'] <= 300):
            errors.append('tempo must be between 0 and 300 BPM')
        if not (0 <= data['acousticness'] <= 1):
            errors.append('acousticness must be between 0 and 1')
        
        if errors:
            return jsonify({
                'success': False,
                'error': 'Invalid feature values',
                'validation_errors': errors
            }), 400
        
        # Convert to numpy array and scale
        features_array = np.array([features])
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        predicted_score = gb_model.predict(features_scaled)[0]
        
        # Ensure score is not negative
        predicted_score = max(0, predicted_score)
        
        # Calculate confidence intervals
        mae = 41.95  # From model evaluation
        lower_bound = max(0, predicted_score - mae)
        upper_bound = predicted_score + mae
        
        # Classify the prediction
        classification = classify_score(predicted_score)
        
        # Generate interpretation
        if predicted_score < 50:
            interpretation = "Expected to receive very few votes. Unlikely to advance far."
        elif predicted_score < 150:
            interpretation = "Poor expected performance. Might not qualify from semi-final."
        elif predicted_score < 250:
            interpretation = "Average expected performance. Likely to get some votes but not highly competitive."
        elif predicted_score < 350:
            interpretation = "Good expected performance. Should be competitive in the contest."
        elif predicted_score < 450:
            interpretation = "Excellent expected performance. Strong contender for top rankings."
        else:
            interpretation = "Exceptional performance expected. Potential winner or record-breaker!"
        
        # Return prediction response
        return jsonify({
            'success': True,
            'predicted_score': round(predicted_score, 2),
            'classification': classification,
            'confidence_level': '68%',
            'mae_margin': f'±{mae:.2f}',
            'prediction_range': f'{round(lower_bound, 2)}-{round(upper_bound, 2)}',
            'prediction_range_explanation': f'95% of predictions within ±{2*mae:.2f}',
            'interpretation': interpretation,
            'input_features': {
                'danceability': data['danceability'],
                'energy': data['energy'],
                'valence': data['valence'],
                'tempo': data['tempo'],
                'acousticness': data['acousticness']
            },
            'model_info': {
                'model_name': 'Gradient Boosting Regressor',
                'r_squared': 0.3062,
                'accuracy': 'Moderate'
            },
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': f'Invalid data type: {str(e)}',
            'hint': 'Ensure all numeric fields are numbers'
        }), 400
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

# ============================================================
# BATCH PREDICTION ENDPOINT
# ============================================================

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict scores for multiple songs at once
    
    Expected JSON input:
    {
        "songs": [
            {
                "song_name": "Song 1",
                "danceability": 0.7,
                "energy": 0.8,
                "valence": 0.6,
                "tempo": 120,
                "acousticness": 0.1
            },
            {
                "song_name": "Song 2",
                "danceability": 0.5,
                "energy": 0.6,
                "valence": 0.7,
                "tempo": 100,
                "acousticness": 0.2
            }
        ]
    }
    """
    
    try:
        data = request.get_json()
        
        if 'songs' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing "songs" key in request'
            }), 400
        
        songs = data['songs']
        predictions = []
        errors = []
        
        for idx, song in enumerate(songs):
            try:
                # Extract features
                features = []
                for feature in feature_names:
                    if feature not in song:
                        errors.append(f"Song {idx}: Missing feature '{feature}'")
                        continue
                    features.append(song[feature])
                
                if len(features) != len(feature_names):
                    continue
                
                # Scale and predict
                features_array = np.array([features])
                features_scaled = scaler.transform(features_array)
                predicted_score = max(0, gb_model.predict(features_scaled)[0])
                
                mae = 41.95
                predictions.append({
                    'song_name': song.get('song_name', f'Song {idx+1}'),
                    'predicted_score': round(predicted_score, 2),
                    'classification': classify_score(predicted_score),
                    'range': f'{round(max(0, predicted_score - mae), 2)}-{round(predicted_score + mae, 2)}'
                })
            
            except Exception as e:
                errors.append(f"Song {idx}: {str(e)}")
        
        return jsonify({
            'success': True,
            'total_songs': len(songs),
            'successful_predictions': len(predictions),
            'predictions': predictions,
            'errors': errors if errors else None,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Batch prediction error: {str(e)}'
        }), 500

# ============================================================
# MODEL INFO ENDPOINT
# ============================================================

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the model"""
    return jsonify({
        'model_name': 'Gradient Boosting Regressor',
        'features': feature_names,
        'test_mae': 41.95,
        'r_squared': 0.3062,
        'accuracy': 'Moderate',
        'description': 'Predicts Eurovision song contest scores based on audio features',
        'feature_descriptions': {
            'danceability': 'How suitable for dancing (0-1)',
            'energy': 'Intensity and activity level (0-1)',
            'valence': 'Musical positiveness/happiness (0-1)',
            'tempo': 'Beats per minute',
            'acousticness': 'Likelihood of being acoustic (0-1)'
        },
        'endpoints': {
            '/predict': 'POST - Single song prediction',
            '/predict_batch': 'POST - Multiple songs prediction',
            '/model_info': 'GET - Model information',
            '/health': 'GET - Health check'
        }
    }), 200

# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': {
            '/predict': 'POST',
            '/predict_batch': 'POST',
            '/model_info': 'GET',
            '/health': 'GET'
        }
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'success': False,
        'error': 'Method not allowed'
    }), 405

# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == '__main__':
    print("="*80)
    print("EUROVISION PREDICTION API - FLASK APPLICATION")
    print("="*80)
    print("\nStarting Flask server...")
    print("\nAvailable endpoints:")
    print("  POST /predict - Single song prediction")
    print("  POST /predict_batch - Multiple songs prediction")
    print("  GET  /model_info - Model information")
    print("  GET  /health - Health check")
    print("\n" + "="*80)
    print("Server running on http://localhost:5000")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)