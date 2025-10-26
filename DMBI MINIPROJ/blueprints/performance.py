from flask import Blueprint, render_template
import joblib

performance_bp = Blueprint('performance', __name__)

@performance_bp.route('/performance')
def performance_page():
    """Model Performance page"""
    try:
        # Load model metrics
        metrics = joblib.load('models/metrics.pkl')
        
        # Format metrics for display
        model_metrics = {
            'classification_accuracy': round(metrics.get('classification_accuracy', 0) * 100, 1),
            'regression_rmse': round(metrics.get('regression_rmse', 0), 2),
            'regression_mae': round(metrics.get('regression_mae', 0), 2),
            'r2_score': round(metrics.get('r2_score', 0), 3)
        }
        
        # Additional performance insights
        performance_insights = [
            {
                'title': 'Classification Model',
                'description': 'Random Forest classifier for AQI category prediction',
                'accuracy': model_metrics['classification_accuracy'],
                'status': 'Excellent' if model_metrics['classification_accuracy'] > 85 else 'Good'
            },
            {
                'title': 'Regression Model',
                'description': 'Linear regression for precise AQI value prediction',
                'rmse': model_metrics['regression_rmse'],
                'status': 'Good' if model_metrics['regression_rmse'] < 20 else 'Needs Improvement'
            }
        ]
        
        return render_template('performance.html', 
                             metrics=model_metrics, 
                             insights=performance_insights)
    
    except Exception as e:
        # Fallback metrics if not available
        model_metrics = {
            'classification_accuracy': 0,
            'regression_rmse': 0,
            'regression_mae': 0,
            'r2_score': 0
        }
        performance_insights = []
        
        return render_template('performance.html', 
                             metrics=model_metrics, 
                             insights=performance_insights)