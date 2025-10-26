from flask import Blueprint, render_template
import joblib

patterns_bp = Blueprint('patterns', __name__)

@patterns_bp.route('/patterns')
def patterns_page():
    """Pattern Discovery page showing association rules"""
    try:
        # Load association rules
        association_rules = joblib.load('models/association_rules.pkl')
        
        # Calculate summary statistics
        if association_rules:
            avg_confidence = sum(rule.get('confidence', 0) for rule in association_rules) / len(association_rules)
            max_lift = max(rule.get('lift', 0) for rule in association_rules)
            
            # Process rules for better display
            processed_rules = []
            for i, rule in enumerate(association_rules):
                processed_rule = {
                    'id': i + 1,
                    'rule': rule.get('rule', 'N/A'),
                    'support': round(rule.get('support', 0), 3),
                    'confidence': round(rule.get('confidence', 0), 3),
                    'lift': round(rule.get('lift', 0), 2),
                    'recommendation': generate_recommendation(rule.get('rule', ''))
                }
                processed_rules.append(processed_rule)
        else:
            avg_confidence = 0
            max_lift = 0
            processed_rules = []
        
        summary_stats = {
            'total_patterns': len(processed_rules),
            'avg_confidence': round(avg_confidence, 2),
            'max_lift': round(max_lift, 2)
        }
        
        return render_template('patterns.html', 
                             rules=processed_rules, 
                             summary_stats=summary_stats)
    
    except Exception as e:
        # Fallback if no rules available
        return render_template('patterns.html', 
                             rules=[], 
                             summary_stats={'total_patterns': 0, 'avg_confidence': 0, 'max_lift': 0})

def generate_recommendation(rule_text):
    """Generate actionable recommendation based on rule"""
    if 'High' in rule_text and 'Traffic' in rule_text:
        return 'Implement traffic reduction measures during peak hours'
    elif 'High' in rule_text and 'PM' in rule_text:
        return 'Increase air quality monitoring and implement emission controls'
    elif 'Unhealthy' in rule_text:
        return 'Issue health advisories and recommend indoor activities'
    elif 'Industrial' in rule_text:
        return 'Strengthen industrial emission regulations and monitoring'
    else:
        return 'Monitor conditions and implement preventive measures'