from flask import Blueprint, render_template, request, jsonify
import numpy as np
import joblib

predict_bp = Blueprint('predict', __name__)

@predict_bp.route('/predict')
def predict_page():
    """AI Predictions page"""
    return render_template('predict.html')

@predict_bp.route('/api/predict', methods=['POST'])
def predict_aqi():
    """API endpoint for AQI prediction"""
    try:
        # Load models
        clf_model = joblib.load('models/classification_model.pkl')
        reg_model = joblib.load('models/regression_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        
        data = request.json
        
        # Extract features
        features = [
            float(data.get('pm25', 50)),
            float(data.get('pm10', 80)),
            float(data.get('no2', 40)),
            float(data.get('so2', 15)),
            float(data.get('co', 1.2)),
            float(data.get('o3', 60)),
            float(data.get('traffic', 1000)),
            float(data.get('temperature', 28)),
            float(data.get('humidity', 65)),
            float(data.get('pm25', 50)) / float(data.get('pm10', 80)),  # PM_Ratio
            (float(data.get('pm25', 50)) + float(data.get('pm10', 80)) + float(data.get('no2', 40))) / 3,  # Pollution_Index
            int(data.get('month', 6)),
            int(data.get('dayofweek', 1))
        ]
        
        # Normalize pollutant features (first 6 features)
        features_array = np.array(features).reshape(1, -1)
        features_normalized = features_array.copy()
        features_normalized[0, :6] = scaler.transform(features_array[:, :6].reshape(1, -1))[0]
        
        # Make predictions
        aqi_category = clf_model.predict(features_normalized)[0]
        aqi_value = reg_model.predict(features_normalized)[0]
        
        # Get health recommendation
        health_recommendation = get_health_recommendation(aqi_category)
        
        return jsonify({
            'success': True,
            'aqi_category': aqi_category,
            'aqi_value': round(float(aqi_value), 2),
            'health_recommendation': health_recommendation
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

def get_health_recommendation(category):
    """Get health recommendation based on AQI category"""
    recommendations = {
        'Good': 'Air quality is satisfactory. Enjoy outdoor activities!',
        'Moderate': 'Air quality is acceptable for most people. Sensitive individuals should limit prolonged outdoor exertion.',
        'Unhealthy for Sensitive Groups': 'Sensitive groups should reduce outdoor activities and consider wearing masks.',
        'Unhealthy': 'Everyone should limit outdoor activities. Wear masks when going outside.',
        'Very Unhealthy': 'Avoid outdoor activities. Stay indoors and use air purifiers if available.'
    }
    return recommendations.get(category, 'Monitor air quality updates regularly.')