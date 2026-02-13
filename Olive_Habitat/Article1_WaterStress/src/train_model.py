import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def load_data(filepath):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

def train_models(df):
    """Trains Random Forest and XGBoost to predict water stress."""
    
    # 1. Prepare Data
    X = df.drop(columns=['Is_Stressed', 'NDVI', 'NDWI'])
    y = df['Is_Stressed']
    
    # Time Series Split
    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    # Calculate Class Imbalance Weights
    # If 90% are healthy and 10% stressed, we give 9x more weight to stressed examples
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos
    print(f"Training set: {len(X_train)} days. Test set: {len(X_test)} days.")
    print(f"Class Balance: {neg} Healthy, {pos} Stressed. Weight Multiplier: {scale_pos_weight:.2f}")
    
    # 2. Random Forest (with class_weight='balanced')
    # ----------------
    print("\nTraining Random Forest (Balanced)...")
    # class_weight='balanced' automatically adjusts weights inversely proportional to class frequencies
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    
    y_pred_rf = rf.predict(X_test)
    print("Random Forest Results:")
    print(classification_report(y_test, y_pred_rf))
    print(f"ROC-AUC: {roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]):.3f}")

    # 3. XGBoost (with scale_pos_weight)
    # ----------
    print("\nTraining XGBoost (Balanced)...")
    # scale_pos_weight helps the model focus on the minority class
    model_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
    model_xgb.fit(X_train, y_train)
    
    y_pred_xgb = model_xgb.predict(X_test)
    print("XGBoost Results:")
    print(classification_report(y_test, y_pred_xgb))
    print(f"ROC-AUC: {roc_auc_score(y_test, model_xgb.predict_proba(X_test)[:, 1]):.3f}")
    
    return rf, model_xgb, X_test, y_test

def plot_importance(model, feature_names, output_path):
    """Plots feature importance to see WHAT causes stress."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance (What drives Water Stress?)")
    plt.bar(range(X_train.shape[1]), importances[indices], align="center")
    plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    input_csv = "Olive_Habitat/Article1_WaterStress/data/processed_features.csv"
    model_dir = "Olive_Habitat/Article1_WaterStress/models"
    os.makedirs(model_dir, exist_ok=True)
    
    print("Loading processed data...")
    df = load_data(input_csv)
    
    # Train
    rf_model, xgb_model, X_test, y_test = train_models(df)
    
    # Save Models
    joblib.dump(rf_model, os.path.join(model_dir, "rf_model.pkl"))
    joblib.dump(xgb_model, os.path.join(model_dir, "xgb_model.pkl"))
    print(f"Models saved to {model_dir}")
    
    # Feature Importance (Using Random Forest for interpretability)
    # This answers the "Why?" question for the article.
    # Note: We need feature names, reusing X_test.columns
    # plot_importance(rf_model, X_test.columns, os.path.join(model_dir, "feature_importance.png"))

if __name__ == "__main__":
    main()
