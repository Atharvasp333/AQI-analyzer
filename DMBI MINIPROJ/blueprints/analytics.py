from flask import Blueprint, render_template, request, jsonify
import pandas as pd
import numpy as np

analytics_bp = Blueprint('analytics', __name__)

@analytics_bp.route('/analytics')
def analytics_page():
    """Analytics Dashboard page using Google Charts"""
    try:
        df = pd.read_csv('data/processed_air_quality_data.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        
        cities = df['City'].unique().tolist()
        date_range = {
            'min': df['Date'].min().strftime('%Y-%m-%d'),
            'max': df['Date'].max().strftime('%Y-%m-%d')
        }
        
        return render_template('analytics.html', cities=cities, date_range=date_range)
    except Exception as e:
        return render_template('analytics.html', cities=[], date_range={'min': '', 'max': ''})

@analytics_bp.route('/api/analytics')
def get_analytics_data():
    """API endpoint for analytics data"""
    try:
        df = pd.read_csv('data/processed_air_quality_data.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        
        city = request.args.get('city', 'All')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # Filter data
        filtered_df = df.copy()
        
        if city != 'All':
            filtered_df = filtered_df[filtered_df['City'] == city]
        
        if start_date:
            filtered_df = filtered_df[filtered_df['Date'] >= start_date]
        
        if end_date:
            filtered_df = filtered_df[filtered_df['Date'] <= end_date]
        
        # AQI category distribution
        category_counts = filtered_df['AQI_Category'].value_counts().to_dict()
        
        # AQI trend over time (limit to last 30 days for better visualization)
        daily_aqi = filtered_df.groupby('Date')['AQI'].mean().reset_index()
        daily_aqi = daily_aqi.tail(30)  # Last 30 days
        daily_aqi['Date'] = daily_aqi['Date'].dt.strftime('%Y-%m-%d')
        
        # City-wise AQI comparison
        city_aqi = filtered_df.groupby('City')['AQI'].mean().round(1).to_dict()
        
        # Pollutant correlations
        pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        correlations = filtered_df[pollutant_cols + ['AQI']].corr()['AQI'].drop('AQI').round(3).to_dict()
        
        # Monthly trend
        filtered_df['Month'] = filtered_df['Date'].dt.month
        monthly_trend = filtered_df.groupby('Month')['AQI'].mean().round(1).to_dict()
        
        # Statistics
        stats = {
            'total_records': len(filtered_df),
            'avg_aqi': round(filtered_df['AQI'].mean(), 1),
            'max_aqi': round(filtered_df['AQI'].max(), 1),
            'min_aqi': round(filtered_df['AQI'].min(), 1)
        }
        
        return jsonify({
            'success': True,
            'category_distribution': category_counts,
            'aqi_trend': daily_aqi.to_dict('records'),
            'city_comparison': city_aqi,
            'pollutant_correlations': correlations,
            'monthly_trend': monthly_trend,
            'statistics': stats
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })