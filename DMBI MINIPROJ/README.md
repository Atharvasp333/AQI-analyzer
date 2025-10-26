# Air Quality Intelligence Hub

## 🌍 Overview
A modern, AI-powered dashboard for air quality monitoring and prediction in smart cities. Built with Flask and Google Charts, featuring machine learning models for accurate AQI forecasting and pattern discovery.

## ✨ Key Features

### 🤖 AI-Powered Predictions
- **Random Forest Classification** - Predicts AQI categories with 85%+ accuracy
- **Linear Regression** - Provides precise numerical AQI values
- **Real-time Forecasting** - Instant predictions with interactive visualizations
- **Health Recommendations** - Actionable advice based on predictions

### 📊 Interactive Analytics Dashboard
- **AQI Trend Analysis** - Time-series visualization with Google Charts
- **City Comparison** - Multi-city performance analysis
- **Pollutant Correlations** - Identify key pollution factors
- **Category Distribution** - Visual breakdown of air quality levels

### 🔍 Pattern Discovery
- **Association Rule Mining** - Apriori algorithm for hidden patterns
- **Smart Recommendations** - AI-generated improvement strategies
- **Interactive Tables** - Support, confidence, and lift metrics

### 📈 Model Performance Monitoring
- **Real-time Metrics** - Classification accuracy and regression performance
- **Visual Comparisons** - Professional charts showing model effectiveness
- **Performance Insights** - Strengths and improvement areas

## 🚀 Technology Stack

### Backend
- **Flask** - Modern Python web framework with blueprints
- **Pandas & NumPy** - Data processing and numerical computing
- **Scikit-learn** - Machine learning algorithms
- **Joblib** - Model serialization and persistence

### Frontend
- **Google Charts** - Professional, reliable data visualizations
- **Bootstrap 5** - Responsive, modern UI framework
- **Inter Font** - Clean, professional typography
- **Custom CSS** - Minimalist design system

### Machine Learning
- **Random Forest** - AQI category classification
- **Linear Regression** - Continuous AQI value prediction
- **Apriori Algorithm** - Association rule mining
- **StandardScaler** - Feature normalization

## 🛠️ Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd Air-Quality-Intelligence-Hub/DMBI\ MINIPROJ

# Install dependencies
pip install -r requirements.txt

# Run data preprocessing (first time only)
python data_preprocessing.py

# Generate association rules (first time only)
python association_rules.py

# Start the application
python app.py
```

### Access the Dashboard
Open your browser and navigate to: **http://127.0.0.1:5000**

## 📁 Project Structure

```
Air-Quality-Intelligence-Hub/DMBI MINIPROJ/
├── app.py                          # Main Flask application
├── blueprints/                     # Modular route handlers
│   ├── home.py                     # Home page controller
│   ├── predict.py                  # AI predictions
│   ├── analytics.py                # Analytics dashboard
│   ├── patterns.py                 # Pattern discovery
│   ├── performance.py              # Model performance
│   └── about.py                    # Project information
├── templates/                      # HTML templates
│   ├── base.html                   # Base template with navigation
│   ├── home.html                   # Landing page
│   ├── predict.html                # AI predictions with charts
│   ├── analytics.html              # Google Charts dashboard
│   ├── patterns.html               # Association rules
│   ├── performance.html            # Model metrics
│   └── about.html                  # Project documentation
├── static/
│   ├── css/style.css              # Modern minimalist styles
│   └── images/                     # Static assets
├── data/
│   ├── air_quality_data.csv       # Raw dataset
│   └── processed_air_quality_data.csv  # Processed data
├── models/                         # Trained ML models
│   ├── classification_model.pkl   # Random Forest classifier
│   ├── regression_model.pkl       # Linear regression model
│   ├── scaler.pkl                 # Feature scaler
│   ├── metrics.pkl                # Performance metrics
│   └── association_rules.pkl      # Discovered patterns
├── data_preprocessing.py           # Data processing & model training
├── association_rules.py            # Pattern mining
└── requirements.txt                # Python dependencies
```

## 🎯 Usage Guide

### 1. **Home Page** (`/`)
- Project overview and navigation
- Summary statistics and feature cards
- Quick access to all modules

### 2. **AI Predictions** (`/predict`)
- Interactive input form for environmental parameters
- Real-time pollutant visualization chart
- AQI gauge and health recommendations
- Quick preset buttons for different scenarios

### 3. **Analytics Dashboard** (`/analytics`)
- Interactive filters (city, date range)
- Multiple Google Charts visualizations
- Summary statistics cards
- Real-time data updates

### 4. **Pattern Discovery** (`/patterns`)
- Association rules table with metrics
- Color-coded recommendations
- Action plan summary
- Pattern insights and suggestions

### 5. **Model Performance** (`/performance`)
- Performance metrics visualization
- Model configuration details
- Strengths and improvement areas
- Professional performance charts

### 6. **About** (`/about`)
- Technology stack information
- Project objectives and features
- System architecture overview
- Contact and support information

## 📊 Data Features

### Environmental Parameters
- **PM2.5, PM10** - Particulate matter concentrations
- **NO2, SO2, CO, O3** - Gas pollutant levels
- **Temperature, Humidity** - Weather conditions
- **Traffic Volume** - Vehicle density
- **Temporal Features** - Date/time patterns

### Machine Learning Models
- **Classification**: 85%+ accuracy for AQI categories
- **Regression**: RMSE < 20, R² > 0.8 for AQI values
- **Association Rules**: Pattern discovery with confidence metrics

## 🔧 API Endpoints

### Prediction API
```bash
POST /api/predict
Content-Type: application/json

{
    "pm25": 50.0, "pm10": 80.0, "no2": 40.0,
    "so2": 15.0, "co": 1.2, "o3": 60.0,
    "traffic": 1000, "temperature": 28.0, "humidity": 65.0
}
```

### Analytics API
```bash
GET /api/analytics?city=All&start_date=2023-01-01&end_date=2023-12-31
```

## 🎨 Design System

### Colors
- **Primary**: #2563eb (Blue)
- **Success**: #10b981 (Green)
- **Warning**: #f59e0b (Amber)
- **Danger**: #ef4444 (Red)

### Typography
- **Font**: Inter (Google Fonts)
- **Hierarchy**: Bold titles, medium labels, regular body text

### Components
- **Cards**: White background, subtle shadows, 12px radius
- **Charts**: Google Charts with consistent color palette
- **Forms**: Clean inputs with focus states
- **Navigation**: Sidebar with active state indicators

## 🚀 Deployment

### Development
```bash
python app.py  # Runs on http://127.0.0.1:5000
```

### Production Considerations
- Use a production WSGI server (Gunicorn, uWSGI)
- Set up environment variables for configuration
- Implement proper logging and monitoring
- Consider database integration for larger datasets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Google Charts** - Reliable, professional visualizations
- **Scikit-learn** - Machine learning algorithms
- **Flask** - Lightweight, flexible web framework
- **Bootstrap** - Responsive UI components

---

**Built with ❤️ for smart city environmental monitoring**