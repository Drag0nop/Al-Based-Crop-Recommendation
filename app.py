from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import os

# Import your real dataset model
from model import AdvancedCropRecommendationModel  

app = Flask(__name__)
CORS(app)

# Crop profit data (same as before)
CROP_PROFIT_DATA = {
    'rice': {'cost_per_ha': 45000, 'yield_per_ha': 4500, 'price_per_kg': 25, 'season': 'Kharif', 'growth_period': '120-150 days'},
    'wheat': {'cost_per_ha': 40000, 'yield_per_ha': 3500, 'price_per_kg': 30, 'season': 'Rabi', 'growth_period': '110-130 days'},
    'corn': {'cost_per_ha': 35000, 'yield_per_ha': 5000, 'price_per_kg': 20, 'season': 'Kharif', 'growth_period': '90-120 days'},
    'cotton': {'cost_per_ha': 60000, 'yield_per_ha': 500, 'price_per_kg': 80, 'season': 'Kharif', 'growth_period': '160-200 days'},
    'sugarcane': {'cost_per_ha': 80000, 'yield_per_ha': 50000, 'price_per_kg': 3.5, 'season': 'Annual', 'growth_period': '300-365 days'},
    'tomato': {'cost_per_ha': 70000, 'yield_per_ha': 25000, 'price_per_kg': 15, 'season': 'Rabi/Summer', 'growth_period': '90-120 days'},
    'potato': {'cost_per_ha': 55000, 'yield_per_ha': 20000, 'price_per_kg': 12, 'season': 'Rabi', 'growth_period': '90-120 days'},
    'onion': {'cost_per_ha': 50000, 'yield_per_ha': 15000, 'price_per_kg': 18, 'season': 'Rabi', 'growth_period': '120-150 days'}
}

# Initialize and train/load model
DATASET_PATH = "Crop_recommendation.csv"
MODEL_PATH = "crop_recommendation_model.pkl"

if os.path.exists(MODEL_PATH):
    crop_model = AdvancedCropRecommendationModel(DATASET_PATH)
    crop_model.load_model(MODEL_PATH)
else:
    crop_model = AdvancedCropRecommendationModel(DATASET_PATH)
    crop_model.train_model()
    crop_model.save_model(MODEL_PATH)

def calculate_profit_analysis(crop_name):
    """Calculate detailed profit analysis for a crop"""
    data = CROP_PROFIT_DATA.get(crop_name.lower())
    if not data:
        return None
    
    revenue = data['yield_per_ha'] * data['price_per_kg']
    profit = revenue - data['cost_per_ha']
    profit_margin = (profit / revenue) * 100 if revenue > 0 else 0
    
    return {
        'crop_name': crop_name,
        'cost_per_ha': data['cost_per_ha'],
        'yield_per_ha': data['yield_per_ha'],
        'price_per_kg': data['price_per_kg'],
        'season': data['season'],
        'growth_period': data['growth_period'],
        'revenue': revenue,
        'profit': profit,
        'profit_margin': round(profit_margin, 1)
    }

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict_crops():
    """API endpoint for crop prediction"""
    try:
        data = request.get_json()
        required_fields = ['nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Format input for model
        input_features = [
            data['nitrogen'],
            data['phosphorus'],
            data['potassium'],
            data['temperature'],
            data['humidity'],
            data['ph'],
            data['rainfall']
        ]
        
        predictions = crop_model.predict_with_confidence(input_features)
        
        # Add profit analysis
        for pred in predictions[:3]:  # keep top 3
            pred['profit_analysis'] = calculate_profit_analysis(pred['crop'])
        
        return jsonify({
            'success': True,
            'predictions': predictions[:3],
            'input_data': data
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/crop-info/<crop_name>')
def get_crop_info(crop_name):
    """Get detailed information about a specific crop"""
    profit_analysis = calculate_profit_analysis(crop_name)
    if profit_analysis:
        return jsonify({'success': True, 'crop_info': profit_analysis})
    else:
        return jsonify({'error': 'Crop not found'}), 404

@app.route('/api/model-info')
def get_model_info():
    """Get information about the trained model"""
    return jsonify({
        'success': True,
        'model_info': {
            'algorithm': 'Random Forest Classifier',
            'features': crop_model.feature_names,
            'supported_crops': crop_model.crop_names.tolist() if crop_model.crop_names is not None else [],
            'model_trained': crop_model.model is not None
        }
    })

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_ready': crop_model.model is not None})

if __name__ == '__main__':
    print("Starting Crop Recommendation System with Real Dataset...")
    app.run(debug=True, host='0.0.0.0', port=5000)
