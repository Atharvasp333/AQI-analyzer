from flask import Flask, render_template, jsonify
from blueprints.home import home_bp
from blueprints.predict import predict_bp
from blueprints.patterns import patterns_bp
from blueprints.analytics import analytics_bp
from blueprints.performance import performance_bp
from blueprints.about import about_bp

app = Flask(__name__)

# Register blueprints
app.register_blueprint(home_bp)
app.register_blueprint(predict_bp)
app.register_blueprint(patterns_bp)
app.register_blueprint(analytics_bp)
app.register_blueprint(performance_bp)
app.register_blueprint(about_bp)

@app.route('/setup')
def setup():
    """Setup page for running preprocessing"""
    return render_template('setup.html')

@app.route('/run_setup', methods=['POST'])
def run_setup():
    """Run data preprocessing and model training"""
    try:
        # Import and run preprocessing
        from data_preprocessing import preprocess_data, train_models
        from association_rules import mine_association_rules
        
        # Run preprocessing
        df = preprocess_data()
        
        # Train models
        clf_model, reg_model, metrics = train_models()
        
        # Mine association rules
        rules = mine_association_rules()
        
        return jsonify({
            'success': True,
            'message': 'Setup completed successfully! Please refresh the page.'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)