import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

class ForestFirePredictor:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
            'ANN': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
            'XGBoost': xgb.XGBRegressor(random_state=42),
            'LightGBM': lgb.LGBMRegressor(random_state=42)
        }
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.best_models = {}
        self.model_params = {
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Decision Tree': {
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Extra Trees': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.01, 0.1, 0.3]
            },
            'LightGBM': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.01, 0.1, 0.3]
            }
        }
        
    def load_data(self, filepath):
        """Load and preprocess the dataset."""
        try:
            data = pd.read_csv(filepath)
            return data
        except FileNotFoundError:
            print(f"Error: Dataset file not found at {filepath}")
            return None
            
    def prepare_data(self, data):
        """Prepare data for training and testing."""
        # Create a copy of the data to avoid modifying the original
        df = data.copy()
        
        # Encode categorical variables
        categorical_columns = ['month', 'day']
        for column in categorical_columns:
            self.label_encoders[column] = LabelEncoder()
            df[column] = self.label_encoders[column].fit_transform(df[column])
        
        # Separate features and target
        X = df.drop('area', axis=1)
        y = df['area']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns
    
    def perform_cross_validation(self, model, X, y, cv=5):
        """Perform cross-validation on the model."""
        scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
        return np.sqrt(-scores.mean())
    
    def tune_hyperparameters(self, model_name, model, X_train, y_train):
        """Tune hyperparameters using GridSearchCV."""
        if model_name not in self.model_params:
            return model
            
        grid_search = GridSearchCV(
            model,
            self.model_params[model_name],
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        print(f"\nBest parameters for {model_name}:")
        print(grid_search.best_params_)
        return grid_search.best_estimator_
    
    def train_models(self, X_train, y_train):
        """Train all models with cross-validation and hyperparameter tuning."""
        trained_models = {}
        cv_scores = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Perform cross-validation
            cv_score = self.perform_cross_validation(model, X_train, y_train)
            cv_scores[name] = cv_score
            print(f"Cross-validation RMSE: {cv_score:.4f}")
            
            # Tune hyperparameters
            best_model = self.tune_hyperparameters(name, model, X_train, y_train)
            
            # Ensure the model is fitted
            if name == 'ANN':
                best_model.fit(X_train, y_train)
            
            trained_models[name] = best_model
            
        return trained_models, cv_scores
    
    def evaluate_models(self, trained_models, X_test, y_test):
        """Evaluate all models and return their performance metrics."""
        results = {}
        for name, model in trained_models.items():
            y_pred = model.predict(X_test)
            results[name] = {
                'MSE': mean_squared_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'R2': r2_score(y_test, y_pred),
                'MAE': mean_absolute_error(y_test, y_pred)
            }
        return results
    
    def plot_feature_importance(self, model, feature_names):
        """Plot feature importance for tree-based models."""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            plt.figure(figsize=(10, 6))
            sns.barplot(x=importance, y=feature_names)
            plt.title('Feature Importance')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.show()
    
    def plot_model_comparison(self, results):
        """Plot comparison of model performances."""
        metrics = ['MSE', 'RMSE', 'R2', 'MAE']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics):
            values = [results[model][metric] for model in results]
            sns.barplot(x=list(results.keys()), y=values, ax=axes[idx])
            axes[idx].set_title(f'{metric} Comparison')
            axes[idx].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def save_models(self, models, scaler, label_encoders, feature_names):
        """Save trained models and preprocessing objects."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"models_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Save models
        for name, model in models.items():
            joblib.dump(model, f"{save_dir}/{name.lower().replace(' ', '_')}.joblib")
        
        # Save preprocessing objects
        joblib.dump(scaler, f"{save_dir}/scaler.joblib")
        joblib.dump(label_encoders, f"{save_dir}/label_encoders.joblib")
        joblib.dump(feature_names, f"{save_dir}/feature_names.joblib")
        
        print(f"\nModels and preprocessing objects saved in {save_dir}")
    
    def load_models(self, model_dir):
        """Load trained models and preprocessing objects."""
        # Load models
        for name in self.models.keys():
            model_path = f"{model_dir}/{name.lower().replace(' ', '_')}.joblib"
            if os.path.exists(model_path):
                self.best_models[name] = joblib.load(model_path)
        
        # Load preprocessing objects
        self.scaler = joblib.load(f"{model_dir}/scaler.joblib")
        self.label_encoders = joblib.load(f"{model_dir}/label_encoders.joblib")
        feature_names = joblib.load(f"{model_dir}/feature_names.joblib")
        
        print(f"\nModels and preprocessing objects loaded from {model_dir}")
        return feature_names

def main():
    # Initialize the predictor
    predictor = ForestFirePredictor()
    
    # Load data
    data = predictor.load_data('data/forestfires.csv')
    if data is None:
        return
    
    # Prepare data
    X_train, X_test, y_train, y_test, feature_names = predictor.prepare_data(data)
    
    # Train models with cross-validation and hyperparameter tuning
    trained_models, cv_scores = predictor.train_models(X_train, y_train)
    
    # Evaluate models
    results = predictor.evaluate_models(trained_models, X_test, y_test)
    
    # Print results
    print("\nModel Performance Results:")
    for model, metrics in results.items():
        print(f"\n{model}:")
        print(f"Cross-validation RMSE: {cv_scores[model]:.4f}")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    # Plot feature importance for Random Forest
    predictor.plot_feature_importance(trained_models['Random Forest'], feature_names)
    
    # Plot model comparison
    predictor.plot_model_comparison(results)
    
    # Save models and preprocessing objects
    predictor.save_models(trained_models, predictor.scaler, predictor.label_encoders, feature_names)

if __name__ == "__main__":
    main() 