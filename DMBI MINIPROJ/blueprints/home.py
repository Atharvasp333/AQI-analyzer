from flask import Blueprint, render_template
import pandas as pd

home_bp = Blueprint('home', __name__)

@home_bp.route('/')
def index():
    """Home page with overview and navigation"""
    try:
        # Load basic stats for summary cards
        df = pd.read_csv('data/processed_air_quality_data.csv')
        
        stats = {
            'total_cities': len(df['City'].unique()),
            'total_records': len(df),
            'date_range': {
                'start': df['Date'].min(),
                'end': df['Date'].max()
            }
        }
        
        # Calculate model accuracy (placeholder - will be loaded from metrics)
        model_accuracy = 85.2  # This should come from metrics.pkl
        
        return render_template('home.html', stats=stats, model_accuracy=model_accuracy)
    except Exception as e:
        # Fallback stats if data not available
        stats = {
            'total_cities': 0,
            'total_records': 0,
            'date_range': {'start': 'N/A', 'end': 'N/A'}
        }
        return render_template('home.html', stats=stats, model_accuracy=0)