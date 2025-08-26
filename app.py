from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
import joblib
import os
import re
from typing import Dict, List, Any
import uuid

# Import your real dataset model
from model import AdvancedCropRecommendationModel  

app = Flask(__name__)
CORS(app)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

# Crop profit data (same as before)
CROP_PROFIT_DATA = {
    'rice': {'cost_per_ha': 45000, 'yield_per_ha': 4500, 'price_per_kg': 25, 'season': 'Kharif', 'growth_period': '120-150 days'},
    'wheat': {'cost_per_ha': 40000, 'yield_per_ha': 3500, 'price_per_kg': 30, 'season': 'Rabi', 'growth_period': '110-130 days'},
    'corn': {'cost_per_ha': 35000, 'yield_per_ha': 5000, 'price_per_kg': 20, 'season': 'Kharif', 'growth_period': '90-120 days'},
    'cotton': {'cost_per_ha': 60000, 'yield_per_ha': 500, 'price_per_kg': 80, 'season': 'Kharif', 'growth_period': '160-200 days'},
    'sugarcane': {'cost_per_ha': 80000, 'yield_per_ha': 50000, 'price_per_kg': 3.5, 'season': 'Annual', 'growth_period': '300-365 days'},
    'tomato': {'cost_per_ha': 70000, 'yield_per_ha': 25000, 'price_per_kg': 15, 'season': 'Rabi/Summer', 'growth_period': '90-120 days'},
    'potato': {'cost_per_ha': 55000, 'yield_per_ha': 20000, 'price_per_kg': 12, 'season': 'Rabi', 'growth_period': '90-120 days'},
    'onion': {'cost_per_ha': 50000, 'yield_per_ha': 15000, 'price_per_kg': 18, 'season': 'Rabi', 'growth_period': '120-150 days'},
    'barley': {'cost_per_ha': 38000, 'yield_per_ha': 2800, 'price_per_kg': 28, 'season': 'Rabi', 'growth_period': '100-120 days'},
    'mustard': {'cost_per_ha': 30000, 'yield_per_ha': 1200, 'price_per_kg': 50, 'season': 'Rabi', 'growth_period': '100-110 days'},
    'groundnut': {'cost_per_ha': 45000, 'yield_per_ha': 2000, 'price_per_kg': 55, 'season': 'Kharif', 'growth_period': '120-150 days'},
    'soybean': {'cost_per_ha': 42000, 'yield_per_ha': 2200, 'price_per_kg': 45, 'season': 'Kharif', 'growth_period': '90-110 days'},
    'pigeonpea': {'cost_per_ha': 30000, 'yield_per_ha': 1500, 'price_per_kg': 70, 'season': 'Kharif', 'growth_period': '150-210 days'},
    'chickpea': {'cost_per_ha': 28000, 'yield_per_ha': 1800, 'price_per_kg': 60, 'season': 'Rabi', 'growth_period': '100-120 days'},
    'banana': {'cost_per_ha': 100000, 'yield_per_ha': 60000, 'price_per_kg': 10, 'season': 'Annual', 'growth_period': '300-365 days'},
    'mango': {'cost_per_ha': 120000, 'yield_per_ha': 10000, 'price_per_kg': 40, 'season': 'Summer', 'growth_period': '3-5 years (orchard crop)'},
    'tea': {'cost_per_ha': 150000, 'yield_per_ha': 2500, 'price_per_kg': 150, 'season': 'Annual', 'growth_period': '4-5 years (perennial)'},
    'coffee': {'cost_per_ha': 180000, 'yield_per_ha': 2000, 'price_per_kg': 200, 'season': 'Annual', 'growth_period': '3-4 years (perennial)'}
}

class CropChatbot:
    def __init__(self, profit_data):
        # Enhanced crop database combining your profit data with additional information
        self.crop_data = {}
        self.init_crop_database(profit_data)
        
    def init_crop_database(self, profit_data):
        """Initialize comprehensive crop database"""
        # Additional crop information to supplement profit data
        crop_details = {
            'rice': {
                "scientific_name": "Oryza sativa",
                "type": "Cereal grain",
                "climate": "Tropical and subtropical, requires high humidity",
                "soil": "Clay or loamy soil with good water retention",
                "water_requirement": "High - requires flooded fields",
                "temperature": "20-35°C optimal",
                "major_diseases": ["Blast", "Brown spot", "Bacterial blight"],
                "pests": ["Rice stem borer", "Brown planthopper", "Rice weevil"],
                "nutrients": "Rich in carbohydrates, provides energy",
                "care_tips": [
                    "Maintain water level 2-5cm in fields",
                    "Apply nitrogen fertilizer in split doses",
                    "Monitor for pest infestations regularly"
                ]
            },
            'wheat': {
                "scientific_name": "Triticum aestivum",
                "type": "Cereal grain",
                "climate": "Temperate climate with cool winters",
                "soil": "Well-drained loamy soil",
                "water_requirement": "Moderate - avoid waterlogging",
                "temperature": "15-25°C optimal",
                "major_diseases": ["Rust", "Powdery mildew", "Septoria leaf blotch"],
                "pests": ["Aphids", "Armyworm", "Hessian fly"],
                "nutrients": "Rich in protein, fiber, and B vitamins",
                "care_tips": [
                    "Ensure proper drainage",
                    "Apply balanced fertilizers",
                    "Practice crop rotation"
                ]
            },
            'tomato': {
                "scientific_name": "Solanum lycopersicum",
                "type": "Vegetable/Fruit",
                "climate": "Warm temperate climate",
                "soil": "Well-drained, fertile soil with pH 6.0-6.8",
                "water_requirement": "Regular watering, avoid overwatering",
                "temperature": "18-25°C optimal",
                "major_diseases": ["Early blight", "Late blight", "Fusarium wilt"],
                "pests": ["Tomato hornworm", "Whitefly", "Aphids"],
                "nutrients": "Rich in lycopene, vitamin C, and potassium",
                "care_tips": [
                    "Provide support for vining varieties",
                    "Mulch to retain moisture",
                    "Prune suckers for better fruit development"
                ]
            },
            'corn': {
                "scientific_name": "Zea mays",
                "type": "Cereal grain",
                "climate": "Warm climate with adequate rainfall",
                "soil": "Well-drained, fertile soil",
                "water_requirement": "Moderate to high",
                "temperature": "21-27°C optimal",
                "major_diseases": ["Corn smut", "Gray leaf spot", "Corn rust"],
                "pests": ["Corn borer", "Fall armyworm", "Corn earworm"],
                "nutrients": "Rich in carbohydrates and dietary fiber",
                "care_tips": [
                    "Plant in blocks for better pollination",
                    "Side-dress with nitrogen fertilizer",
                    "Control weeds early in growth"
                ]
            },
            'cotton': {
                "scientific_name": "Gossypium hirsutum",
                "type": "Fiber crop",
                "climate": "Warm climate with long frost-free period",
                "soil": "Deep, well-drained black cotton soil",
                "water_requirement": "Moderate to high",
                "temperature": "21-30°C optimal",
                "major_diseases": ["Fusarium wilt", "Verticillium wilt", "Cotton leaf curl virus"],
                "pests": ["Bollworm", "Aphids", "Thrips", "Whitefly"],
                "nutrients": "Seeds rich in protein and oil",
                "care_tips": [
                    "Maintain proper plant spacing",
                    "Regular monitoring for bollworm",
                    "Adequate irrigation during flowering"
                ]
            },
            'potato': {
                "scientific_name": "Solanum tuberosum",
                "type": "Vegetable/Tuber",
                "climate": "Cool temperate climate",
                "soil": "Well-drained, sandy loam soil",
                "water_requirement": "Moderate, consistent moisture",
                "temperature": "15-20°C optimal",
                "major_diseases": ["Late blight", "Early blight", "Potato virus Y"],
                "pests": ["Colorado potato beetle", "Aphids", "Wireworms"],
                "nutrients": "Rich in carbohydrates, potassium, and vitamin C",
                "care_tips": [
                    "Hill soil around plants as they grow",
                    "Avoid overwatering to prevent rot",
                    "Harvest when foliage dies back"
                ]
            }
        }
        
        # Merge profit data with detailed crop information
        for crop_name, profit_info in profit_data.items():
            self.crop_data[crop_name] = {
                **profit_info,
                **(crop_details.get(crop_name, {
                    "scientific_name": f"{crop_name.title()} species",
                    "type": "Agricultural crop",
                    "climate": "Varies based on variety",
                    "soil": "Well-drained fertile soil",
                    "water_requirement": "Moderate",
                    "temperature": "Optimal range varies",
                    "major_diseases": ["Common fungal diseases", "Bacterial infections"],
                    "pests": ["Common agricultural pests"],
                    "nutrients": "Nutritional benefits vary",
                    "care_tips": ["Follow good agricultural practices", "Regular monitoring", "Proper fertilization"]
                }))
            }

    def normalize_input(self, user_input: str) -> str:
        """Normalize user input for better matching"""
        return user_input.lower().strip()

    def find_crop(self, user_input: str) -> str:
        """Find crop mentioned in user input"""
        normalized_input = self.normalize_input(user_input)
        
        # Direct crop name matching
        for crop_key in self.crop_data.keys():
            if crop_key in normalized_input:
                return crop_key
        
        # Alternative names matching
        alternatives = {
            "maize": "corn",
            "paddy": "rice",
            "tomatoes": "tomato",
            "potatoes": "potato",
            "onions": "onion"
        }
        
        for alt_name, crop_key in alternatives.items():
            if alt_name in normalized_input and crop_key in self.crop_data:
                return crop_key
        
        return None

    def get_crop_info(self, crop: str, info_type: str = "general") -> dict:
        """Get specific information about a crop"""
        if crop not in self.crop_data:
            return {"error": "Sorry, I don't have information about that crop."}
        
        crop_info = self.crop_data[crop]
        
        if info_type == "general":
            return {
                "type": "general_info",
                "crop_name": crop.title(),
                "scientific_name": crop_info.get('scientific_name', 'N/A'),
                "type_category": crop_info.get('type', 'Agricultural crop'),
                "season": crop_info.get('season', 'N/A'),
                "growth_period": crop_info.get('growth_period', 'N/A'),
                "climate": crop_info.get('climate', 'N/A'),
                "temperature": crop_info.get('temperature', 'N/A'),
                "soil": crop_info.get('soil', 'N/A'),
                "water_requirement": crop_info.get('water_requirement', 'N/A'),
                "nutrients": crop_info.get('nutrients', 'N/A')
            }
        
        elif info_type == "diseases":
            return {
                "type": "diseases",
                "crop_name": crop.title(),
                "diseases": crop_info.get('major_diseases', [])
            }
        
        elif info_type == "pests":
            return {
                "type": "pests",
                "crop_name": crop.title(),
                "pests": crop_info.get('pests', [])
            }
        
        elif info_type == "care":
            return {
                "type": "care_tips",
                "crop_name": crop.title(),
                "care_tips": crop_info.get('care_tips', [])
            }
        
        elif info_type == "profit":
            return {
                "type": "profit_analysis",
                "crop_name": crop.title(),
                "cost_per_ha": crop_info.get('cost_per_ha', 0),
                "yield_per_ha": crop_info.get('yield_per_ha', 0),
                "price_per_kg": crop_info.get('price_per_kg', 0),
                "revenue": crop_info.get('yield_per_ha', 0) * crop_info.get('price_per_kg', 0),
                "profit": (crop_info.get('yield_per_ha', 0) * crop_info.get('price_per_kg', 0)) - crop_info.get('cost_per_ha', 0)
            }
        
        return {"error": "I couldn't find that specific information."}

    def identify_query_type(self, user_input: str) -> str:
        """Identify what type of information user is asking for"""
        normalized_input = self.normalize_input(user_input)
        
        if any(word in normalized_input for word in ["disease", "diseases", "sick", "infection"]):
            return "diseases"
        elif any(word in normalized_input for word in ["pest", "pests", "insect", "bugs"]):
            return "pests"
        elif any(word in normalized_input for word in ["care", "tips", "how to grow", "growing", "maintain"]):
            return "care"
        elif any(word in normalized_input for word in ["profit", "cost", "revenue", "money", "economics", "income"]):
            return "profit"
        elif any(word in normalized_input for word in ["season", "when", "time", "plant"]):
            return "season"
        elif any(word in normalized_input for word in ["soil", "ground", "earth"]):
            return "soil"
        elif any(word in normalized_input for word in ["water", "irrigation", "watering"]):
            return "water"
        elif any(word in normalized_input for word in ["nutrition", "nutrients", "vitamins", "healthy"]):
            return "nutrition"
        else:
            return "general"

    def get_greeting(self) -> dict:
        """Return greeting message"""
        available_crops = list(self.crop_data.keys())
        return {
            "type": "greeting",
            "message": "Welcome to the Agricultural Crop Information Chatbot!",
            "available_crops": available_crops,
            "capabilities": [
                "General information about crops",
                "Growing conditions and seasons", 
                "Common diseases and pests",
                "Care tips and farming practices",
                "Profit analysis and economics",
                "Soil and water requirements"
            ]
        }

    def process_response(self, user_input: str, session_data: dict = None) -> dict:
        """Process user input and return appropriate response"""
        if session_data is None:
            session_data = {}
            
        normalized_input = self.normalize_input(user_input)
        
        # Handle greetings
        if any(greeting in normalized_input for greeting in ["hi", "hello", "hey", "start"]):
            return self.get_greeting()
        
        # Handle exit commands
        if any(exit_word in normalized_input for exit_word in ["bye", "exit", "quit", "goodbye"]):
            return {
                "type": "goodbye",
                "message": "Thank you for using the Crop Information Chatbot! Happy farming!"
            }
        
        # Handle help requests
        if "help" in normalized_input:
            return {
                "type": "help",
                "message": "I can provide information about these crops",
                "available_crops": list(self.crop_data.keys()),
                "example_questions": [
                    "What is rice?",
                    "Rice diseases", 
                    "How to grow tomatoes?",
                    "Cotton pests",
                    "Wheat profit analysis"
                ]
            }
        
        # Find crop in user input
        crop = self.find_crop(user_input)
        
        if crop:
            session_data["current_crop"] = crop
            query_type = self.identify_query_type(user_input)
            return self.get_crop_info(crop, query_type)
        
        # If no crop found, check if user is asking about current crop
        elif session_data.get("current_crop"):
            query_type = self.identify_query_type(user_input)
            return self.get_crop_info(session_data["current_crop"], query_type)
        
        # Default response for unrecognized input
        else:
            return {
                "type": "error",
                "message": "I'm not sure what you're asking about.",
                "available_crops": list(self.crop_data.keys()),
                "suggestion": "Try asking about a specific crop like 'Tell me about rice' or 'Tomato diseases'"
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

# Initialize chatbot
chatbot = CropChatbot(CROP_PROFIT_DATA)

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

@app.route('/chatbot')
def chatbot_page():
    """Serve the chatbot page"""
    return render_template('chatbot.html')

# Original endpoints (unchanged)
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

# New Chatbot endpoints
@app.route('/api/chatbot/message', methods=['POST'])
def chatbot_message():
    """Handle chatbot messages"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Get or create session ID
        session_id = session.get('chatbot_session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
            session['chatbot_session_id'] = session_id
        
        # Get session data
        session_data = session.get('chatbot_data', {})
        
        # Process message
        response = chatbot.process_response(user_message, session_data)
        
        # Update session data
        if 'current_crop' in response or session_data.get('current_crop'):
            session['chatbot_data'] = session_data
        
        return jsonify({
            'success': True,
            'response': response,
            'session_id': session_id
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chatbot/reset', methods=['POST'])
def reset_chatbot():
    """Reset chatbot session"""
    try:
        session.pop('chatbot_session_id', None)
        session.pop('chatbot_data', None)
        
        return jsonify({
            'success': True,
            'message': 'Chatbot session reset successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chatbot/crops')
def get_available_crops():
    """Get list of available crops for chatbot"""
    return jsonify({
        'success': True,
        'crops': list(CROP_PROFIT_DATA.keys())
    })

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'model_ready': crop_model.model is not None,
        'chatbot_ready': True
    })

if __name__ == '__main__':
    print("Starting Crop Recommendation System with Integrated Chatbot...")
    print("Available endpoints:")
    print("- / : Main crop recommendation interface")
    print("- /chatbot : Chatbot interface") 
    print("- /api/predict : Crop prediction")
    print("- /api/chatbot/message : Chatbot messaging")
    app.run(debug=True, host='0.0.0.0', port=5000)