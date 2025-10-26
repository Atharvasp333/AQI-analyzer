from flask import Blueprint, render_template

about_bp = Blueprint('about', __name__)

@about_bp.route('/about')
def about_page():
    """About page with project information"""
    
    tech_stack = [
        {'name': 'Flask', 'description': 'Web framework for Python', 'icon': 'fab fa-python'},
        {'name': 'Chart.js', 'description': 'Interactive charts and visualizations', 'icon': 'fas fa-chart-bar'},
        {'name': 'Bootstrap 5', 'description': 'Responsive UI framework', 'icon': 'fab fa-bootstrap'},
        {'name': 'Scikit-learn', 'description': 'Machine learning algorithms', 'icon': 'fas fa-brain'},
        {'name': 'Pandas', 'description': 'Data manipulation and analysis', 'icon': 'fas fa-table'},
        {'name': 'NumPy', 'description': 'Numerical computing', 'icon': 'fas fa-calculator'}
    ]
    
    features = [
        {
            'title': 'AI-Powered Predictions',
            'description': 'Advanced machine learning models for accurate AQI forecasting',
            'icon': 'fas fa-brain'
        },
        {
            'title': 'Pattern Discovery',
            'description': 'Association rule mining to identify pollution patterns',
            'icon': 'fas fa-search'
        },
        {
            'title': 'Interactive Analytics',
            'description': 'Real-time charts and visualizations for data insights',
            'icon': 'fas fa-chart-line'
        },
        {
            'title': 'Multi-City Monitoring',
            'description': 'Comprehensive air quality tracking across multiple cities',
            'icon': 'fas fa-city'
        }
    ]
    
    project_info = {
        'title': 'Smart City Air Quality Intelligence Hub',
        'version': '2.0',
        'description': 'An advanced AI-powered dashboard for air quality monitoring, prediction, and analysis in smart cities.',
        'objectives': [
            'Provide real-time air quality monitoring and predictions',
            'Identify pollution patterns using machine learning',
            'Support data-driven environmental policy decisions',
            'Enhance public health awareness through accessible visualizations'
        ]
    }
    
    return render_template('about.html', 
                         tech_stack=tech_stack, 
                         features=features, 
                         project_info=project_info)