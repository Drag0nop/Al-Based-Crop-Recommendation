import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

class AdvancedCropRecommendationModel:
    def __init__(self, dataset_path):
        self.model = None
        self.scaler = None
        self.feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        self.dataset_path = dataset_path
        self.crop_names = None   # Will be populated after reading dataset
        
    def load_and_preprocess_data(self):
        """Load dataset and preprocess features + labels"""
        print("Loading dataset...")
        df = pd.read_csv(self.dataset_path)
        
        # Features and labels
        X = df[self.feature_names].values
        y = df['label'].values  # Make sure your dataset has a 'label' column for crops
        
        # Save crop names for reference
        self.crop_names = np.unique(y)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self):
        """Train Random Forest model on real dataset"""
        X_train, X_test, y_train, y_test = self.load_and_preprocess_data()
        
        print("Training Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_acc = self.model.score(X_train, y_train)
        test_acc = self.model.score(X_test, y_test)
        
        print(f"\nTraining Accuracy: {train_acc:.4f}")
        print(f"Testing Accuracy: {test_acc:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        # Classification report
        y_pred = self.model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return self.model
    
    def predict_with_confidence(self, input_features):
        """Predict crop with confidence interval"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model must be trained first!")
        
        input_array = np.array([input_features])
        input_scaled = self.scaler.transform(input_array)
        
        probabilities = self.model.predict_proba(input_scaled)[0]
        
        # Confidence estimation using all trees
        tree_predictions = np.array([
            estimator.predict_proba(input_scaled)[0]
            for estimator in self.model.estimators_
        ])
        confidence_scores = 1 - np.std(tree_predictions, axis=0)
        
        results = []
        for i, crop in enumerate(self.model.classes_):
            results.append({
                "crop": crop,
                "probability": probabilities[i],
                "confidence": confidence_scores[i]
            })
        results.sort(key=lambda x: x['probability'], reverse=True)
        return results
    
    def save_model(self, filepath='crop_recommendation_model.pkl'):
        """Save model & scaler"""
        if self.model is None:
            raise ValueError("No model to save!")
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'crop_names': self.crop_names
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='crop_recommendation_model.pkl'):
        """Load model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.crop_names = model_data['crop_names']
        print(f"Model loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    crop_model = AdvancedCropRecommendationModel("Crop_recommendation.csv")  # path to your dataset
    crop_model.train_model()
    
    # Test prediction
    test_input = [90, 42, 43, 20.8, 82, 6.5, 202]
    results = crop_model.predict_with_confidence(test_input)
    print("\nTop 3 Recommendations:")
    for i, r in enumerate(results[:3]):
        print(f"{i+1}. {r['crop']}: prob={r['probability']:.3f}, conf={r['confidence']:.3f}")
    
    crop_model.save_model()
