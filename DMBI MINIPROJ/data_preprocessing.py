import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
import joblib
import os

def create_sample_data():
    """Create sample air quality data for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic air quality data
    data = {
        'City': np.random.choice(['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata'], n_samples),
        'Date': pd.date_range('2023-01-01', periods=n_samples, freq='D')[:n_samples],
        'PM2.5': np.random.normal(50, 20, n_samples),
        'PM10': np.random.normal(80, 30, n_samples),
        'NO2': np.random.normal(40, 15, n_samples),
        'SO2': np.random.normal(15, 8, n_samples),
        'CO': np.random.normal(1.2, 0.5, n_samples),
        'O3': np.random.normal(60, 25, n_samples),
        'Traffic_Volume': np.random.normal(1000, 300, n_samples),
        'Temperature': np.random.normal(28, 8, n_samples),
        'Humidity': np.random.normal(65, 15, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Ensure positive values
    pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Traffic_Volume']
    for col in pollutant_cols:
        df[col] = np.abs(df[col])
    
    # Calculate AQI based on PM2.5 (simplified)
    def calculate_aqi(pm25):
        if pm25 <= 12:
            return np.random.normal(25, 5)
        elif pm25 <= 35.4:
            return np.random.normal(75, 10)
        elif pm25 <= 55.4:
            return np.random.normal(125, 15)
        elif pm25 <= 150.4:
            return np.random.normal(175, 20)
        else:
            return np.random.normal(250, 30)
    
    df['AQI'] = df['PM2.5'].apply(calculate_aqi)
    df['AQI'] = np.abs(df['AQI'])
    
    # Create AQI categories
    def categorize_aqi(aqi):
        if aqi <= 50:
            return 'Good'
        elif aqi <= 100:
            return 'Moderate'
        elif aqi <= 150:
            return 'Unhealthy for Sensitive Groups'
        elif aqi <= 200:
            return 'Unhealthy'
        else:
            return 'Very Unhealthy'
    
    df['AQI_Category'] = df['AQI'].apply(categorize_aqi)
    
    return df

def preprocess_data():
    """Preprocess the air quality data"""
    print("Creating sample data...")
    df = create_sample_data()
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save raw data
    df.to_csv('data/air_quality_data.csv', index=False)
    print("Sample data saved to data/air_quality_data.csv")
    
    # Handle missing values (simulate some missing values first)
    missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
    df.loc[missing_indices, 'PM2.5'] = np.nan
    
    # Fill missing values with median
    df['PM2.5'].fillna(df['PM2.5'].median(), inplace=True)
    
    # Feature engineering
    df['PM_Ratio'] = df['PM2.5'] / df['PM10']
    df['Pollution_Index'] = (df['PM2.5'] + df['PM10'] + df['NO2']) / 3
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    
    # Normalize pollutant columns
    scaler = StandardScaler()
    pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    df[pollutant_cols] = scaler.fit_transform(df[pollutant_cols])
    
    # Save scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Save processed data
    df.to_csv('data/processed_air_quality_data.csv', index=False)
    print("Processed data saved to data/processed_air_quality_data.csv")
    
    return df

def train_models():
    """Train classification and regression models"""
    print("Loading processed data...")
    df = pd.read_csv('data/processed_air_quality_data.csv')
    
    # Prepare features
    feature_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Traffic_Volume', 
                   'Temperature', 'Humidity', 'PM_Ratio', 'Pollution_Index', 'Month', 'DayOfWeek']
    X = df[feature_cols]
    y_class = df['AQI_Category']
    y_reg = df['AQI']
    
    # Split data
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X, y_class, y_reg, test_size=0.2, random_state=42
    )
    
    # Train classification model
    print("Training classification model...")
    clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_model.fit(X_train, y_class_train)
    
    # Evaluate classification
    y_class_pred = clf_model.predict(X_test)
    class_accuracy = accuracy_score(y_class_test, y_class_pred)
    print(f"Classification Accuracy: {class_accuracy:.3f}")
    
    # Save classification model
    joblib.dump(clf_model, 'models/classification_model.pkl')
    
    # Train regression model
    print("Training regression model...")
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_reg_train)
    
    # Evaluate regression
    y_reg_pred = reg_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    print(f"Regression RMSE: {rmse:.3f}")
    print(f"Regression MAE: {mae:.3f}")
    
    # Save regression model
    joblib.dump(reg_model, 'models/regression_model.pkl')
    
    # Save metrics
    metrics = {
        'classification_accuracy': class_accuracy,
        'regression_rmse': rmse,
        'regression_mae': mae
    }
    joblib.dump(metrics, 'models/metrics.pkl')
    
    print("Models trained and saved successfully!")
    return clf_model, reg_model, metrics

if __name__ == "__main__":
    # Run preprocessing and model training
    df = preprocess_data()
    clf_model, reg_model, metrics = train_models()
    print("Data preprocessing and model training completed!")