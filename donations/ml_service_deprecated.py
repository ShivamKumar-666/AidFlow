import os
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import joblib
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class WeatherData:
    """Data class for weather information"""
    temperature: float
    humidity: float
    city: str
    is_fallback: bool = False


@dataclass
class PredictionResult:
    """Data class for prediction results"""
    freshness_label: str
    freshness_score: int
    confidence: float
    probabilities: Dict[str, float]
    weather_used: Dict
    
    def __getitem__(self, key):
        """Allow dictionary-style access for backward compatibility"""
        return getattr(self, key)
    
    def get(self, key, default=None):
        """Allow .get() method for backward compatibility"""
        return getattr(self, key, default)
    
    def to_dict(self):
        """Convert to dictionary"""
        result = asdict(self)
        return result


class FoodQualityPredictor:
    """
    Machine Learning model for predicting food quality/freshness.
    
    Features:
    - Random Forest classification
    - Weather data integration
    - Categorical encoding
    - Model persistence
    """
    
    # Class constants
    CATEGORICAL_FEATURES = [
        'storage_condition', 'container_type', 'food_type',
        'moisture_type', 'cooking_method', 'texture', 'smell'
    ]
    NUMERICAL_FEATURES = ['storage_time', 'time_since_cooking']
    WEATHER_TIMEOUT = 2  # seconds
    DEFAULT_WEATHER = {'temperature': 25, 'humidity': 65}
    
    def __init__(self, model_dir: Optional[str] = None, auto_train: bool = True):
        """
        Initialize the predictor.
        
        Args:
            model_dir: Directory for model and data files (defaults to script directory)
            auto_train: Whether to auto-train if model doesn't exist
        """
        # Setup paths
        self.base_dir = model_dir or os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.base_dir, 'food_quality_model.pkl')
        self.csv_path = os.path.join(self.base_dir, 'food_data.csv')
        
        # Initialize attributes
        self.model: Optional[RandomForestClassifier] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_cols: List[str] = []
        self.training_metrics: Dict[str, float] = {}
        
        # Load or train model
        if os.path.exists(self.model_path):
            self.load_model()
        elif auto_train:
            logger.warning("Model not found. Training new model...")
            self.train_model()
        else:
            logger.warning("Model not found and auto_train=False. Call train_model() manually.")

    def fetch_weather_data(self, city: str) -> Dict:
        """
        Fetch real-time weather data with robust error handling.
        
        Args:
            city: City name for weather lookup
            
        Returns:
            Dictionary with temperature, humidity, city, and is_fallback
        """
        try:
            url = f"https://wttr.in/{city}?format=j1"
            response = requests.get(url, timeout=self.WEATHER_TIMEOUT)
            response.raise_for_status()
            
            data = response.json()
            current = data['current_condition'][0]
            
            weather = {
                'temperature': float(current['temp_C']),
                'humidity': float(current['humidity']),
                'city': city,
                'is_fallback': False
            }
            logger.info(f"Weather data fetched for {city}: {weather['temperature']}°C, {weather['humidity']}% humidity")
            return weather
            
        except requests.RequestException as e:
            logger.warning(f"Weather API request failed for {city}: {e}. Using defaults.")
        except (KeyError, ValueError, IndexError) as e:
            logger.warning(f"Weather data parsing failed: {e}. Using defaults.")
        except Exception as e:
            logger.error(f"Unexpected error fetching weather: {e}. Using defaults.")
        
        # Fallback to defaults
        return {
            'temperature': self.DEFAULT_WEATHER['temperature'],
            'humidity': self.DEFAULT_WEATHER['humidity'],
            'city': city,
            'is_fallback': True
        }

    def _validate_csv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean CSV data"""
        required_cols = (
            self.NUMERICAL_FEATURES + 
            self.CATEGORICAL_FEATURES + 
            ['freshness_level']
        )
        
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"CSV missing required columns: {missing_cols}")
        
        # Remove rows with missing values
        initial_rows = len(df)
        df = df.dropna()
        removed = initial_rows - len(df)
        
        if removed > 0:
            logger.info(f"Removed {removed} rows with missing values")
        
        if len(df) < 10:
            raise ValueError(f"Insufficient data: only {len(df)} valid rows after cleaning")
        
        return df

    def train_model(self, test_size: float = 0.2, random_state: int = 42) -> Dict[str, float]:
        """
        Train the Random Forest model with validation.
        
        Args:
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with training metrics
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Training data not found at {self.csv_path}")
        
        logger.info(f"Loading training data from {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        df = self._validate_csv(df)
        
        logger.info(f"Training on {len(df)} samples")
        
        # Encode categorical features
        for col in self.CATEGORICAL_FEATURES:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col])
            self.label_encoders[col] = le
            logger.debug(f"Encoded {col}: {len(le.classes_)} unique values")
        
        # Prepare feature columns
        encoded_cols = [c + '_encoded' for c in self.CATEGORICAL_FEATURES]
        self.feature_cols = self.NUMERICAL_FEATURES + encoded_cols
        
        # Prepare features and target
        X = df[self.feature_cols]
        
        le_target = LabelEncoder()
        y = le_target.fit_transform(df['freshness_level'])
        self.label_encoders['freshness_level'] = le_target
        
        logger.info(f"Target classes: {le_target.classes_}")
        
        # Train/test split for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Train model
        logger.info("Training Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1  # Use all CPU cores
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.training_metrics = {
            'accuracy': accuracy,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        logger.info(f"Model accuracy: {accuracy:.2%}")
        logger.info("\nClassification Report:")
        logger.info("\n" + classification_report(
            y_test, y_pred, 
            target_names=le_target.classes_
        ))
        
        # Save model
        self._save_model()
        
        return self.training_metrics

    def _save_model(self):
        """Save model and associated data"""
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_cols': self.feature_cols,
            'training_metrics': self.training_metrics
        }
        
        joblib.dump(model_data, self.model_path)
        logger.info(f"✅ Model saved to {self.model_path}")

    def load_model(self):
        """Load pre-trained model"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            data = joblib.load(self.model_path)
            
            self.model = data['model']
            self.label_encoders = data['label_encoders']
            self.feature_cols = data.get('feature_cols', [])
            self.training_metrics = data.get('training_metrics', {})
            
            logger.info("✅ Model loaded successfully")
            
            if self.training_metrics:
                logger.info(f"Model trained on {self.training_metrics.get('train_samples', 'unknown')} samples")
                logger.info(f"Model accuracy: {self.training_metrics.get('accuracy', 0):.2%}")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _prepare_features(self, input_data: Dict) -> pd.DataFrame:
        """
        Prepare feature vector from input data.
        
        Args:
            input_data: Dictionary with input features
            
        Returns:
            DataFrame with encoded features
        """
        if not self.feature_cols:
            raise RuntimeError("Model not trained or loaded. Feature columns not available.")
        
        # Extract numerical features
        features = []
        for col in self.NUMERICAL_FEATURES:
            if col not in input_data:
                raise ValueError(f"Missing required feature: {col}")
            try:
                features.append(float(input_data[col]))
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid value for {col}: {input_data[col]}") from e
        
        # Encode categorical features
        for col in self.CATEGORICAL_FEATURES:
            val = input_data.get(col)
            
            if val is None:
                raise ValueError(f"Missing required feature: {col}")
            
            le = self.label_encoders.get(col)
            if le is None:
                raise RuntimeError(f"Label encoder not found for {col}")
            
            # Handle unseen values with warning
            if val not in le.classes_:
                logger.warning(
                    f"Unseen value '{val}' for {col}. "
                    f"Known values: {list(le.classes_)}. Using default encoding (0)."
                )
                features.append(0)
            else:
                features.append(le.transform([val])[0])
        
        # Create DataFrame with proper column names
        return pd.DataFrame([features], columns=self.feature_cols)

    def _calculate_freshness_score(
        self, 
        probabilities: np.ndarray, 
        classes: np.ndarray
    ) -> int:
        """
        Calculate freshness score (0-100) from class probabilities.
        
        Fresh = 100 points, Medium = 50 points, Spoiled = 0 points
        """
        score = 0.0
        
        # Map class names to weights
        class_weights = {
            'Fresh': 100,
            'Medium': 50,
            'Spoiled': 0
        }
        
        for cls, prob in zip(classes, probabilities):
            weight = class_weights.get(cls, 0)
            score += prob * weight
        
        return int(round(score))

    def predict(self, input_data: Dict, include_weather: bool = True, return_dict: bool = False):
        """
        Predict food quality/freshness.
        
        Args:
            input_data: Dictionary containing all required features
            include_weather: Whether to fetch weather data
            return_dict: If True, return plain dict instead of PredictionResult (for backward compatibility)
            
        Returns:
            PredictionResult (subscriptable) or dict with label, score, and confidence
            
        Raises:
            ValueError: If required features are missing or invalid
            RuntimeError: If model is not loaded
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() or train_model() first.")
        
        # Fetch weather (optional feature for future use)
        weather = None
        if include_weather:
            city = input_data.get('city', 'Mumbai')
            weather = self.fetch_weather_data(city)
        
        # Prepare features
        X = self._prepare_features(input_data)
        
        # Predict probabilities
        probabilities = self.model.predict_proba(X)[0]
        classes = self.label_encoders['freshness_level'].classes_
        
        # Get prediction
        pred_idx = np.argmax(probabilities)
        predicted_label = classes[pred_idx]
        confidence = probabilities[pred_idx] * 100
        
        # Calculate freshness score
        freshness_score = self._calculate_freshness_score(probabilities, classes)
        
        # Create probability dictionary
        prob_dict = {cls: float(prob) for cls, prob in zip(classes, probabilities)}
        
        # Return dict for backward compatibility
        if return_dict:
            return {
                'freshness_label': predicted_label,
                'freshness_score': freshness_score,
                'confidence': round(confidence, 1),
                'probabilities': prob_dict,
                'weather_used': weather
            }
        
        # Return PredictionResult (subscriptable)
        return PredictionResult(
            freshness_label=predicted_label,
            freshness_score=freshness_score,
            confidence=round(confidence, 1),
            probabilities=prob_dict,
            weather_used=weather
        )

    def predict_batch(self, input_list: List[Dict], return_dict: bool = False):
        """Predict on multiple samples efficiently"""
        return [self.predict(data, return_dict=return_dict) for data in input_list]

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the Random Forest model.
        
        Returns:
            DataFrame with features and their importance scores
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df


# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = FoodQualityPredictor()
    
    # Example prediction
    sample_input = {
        'storage_time': 12,
        'time_since_cooking': 6,
        'storage_condition': 'refrigerated',
        'container_type': 'plastic',
        'food_type': 'cooked_rice',
        'moisture_type': 'moist',
        'cooking_method': 'boiled',
        'texture': 'soft',
        'smell': 'normal',
        'city': 'Mumbai'
    }
    
    try:
        # Test both return types
        result = predictor.predict(sample_input)
        print(f"\n🍽️  Food Quality Prediction")
        print(f"{'='*50}")
        print(f"Freshness: {result.freshness_label}")
        print(f"Score: {result.freshness_score}/100")
        print(f"Confidence: {result.confidence}%")
        
        # Test dictionary-style access (backward compatibility)
        print(f"\nDictionary-style access:")
        print(f"Freshness: {result['freshness_label']}")
        print(f"Score: {result['freshness_score']}/100")
        
        print(f"\nProbabilities:")
        for cls, prob in result.probabilities.items():
            print(f"  {cls}: {prob*100:.1f}%")
        
        if result.weather_used:
            print(f"\nWeather in {result.weather_used['city']}:")
            print(f"  Temperature: {result.weather_used['temperature']}°C")
            print(f"  Humidity: {result.weather_used['humidity']}%")
            if result.weather_used['is_fallback']:
                print("  (Using default values)")
        
        # Show feature importance
        print(f"\n📊 Top 5 Most Important Features:")
        importance = predictor.get_feature_importance().head(5)
        for idx, row in importance.iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
            
    except Exception as e:
        logger.error(f"Prediction failed: {e}")