import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Condition descriptions for the UI
condition_descriptions = {
    "Diabetes": """
    Diabetes is a chronic condition characterized by high blood glucose levels. The primary finding is elevated 
    glucose levels, typically above 126 mg/dL when fasting. Other parameters may show secondary effects of 
    diabetes including electrolyte imbalances and potentially elevated white blood cell counts if infection 
    is present as a complication.
    """,
    
    "Anemia": """
    Anemia is characterized by low hemoglobin levels, typically below 12 g/dL in women and 13.5 g/dL in men.
    This condition results in reduced oxygen-carrying capacity of the blood. Different types of anemia may 
    also show changes in other parameters such as platelet count, depending on the underlying cause.
    """,
    
    "Kidney Disease": """
    Kidney disease is characterized by elevated creatinine and blood urea nitrogen (BUN) levels, indicating 
    reduced kidney function. Electrolyte imbalances such as abnormal sodium and potassium levels may also 
    be present as the kidneys' ability to regulate these electrolytes becomes impaired.
    """,
    
    "Infection": """
    Infectious processes are typically characterized by elevated white blood cell counts (leukocytosis) above 
    11.0 ×10^9/L. Some infections may also cause changes in platelet counts. Depending on the severity and 
    type of infection, other parameters may be affected as secondary effects.
    """,
    
    "Normal": """
    All parameters are within normal reference ranges, suggesting no significant abnormalities in the tested 
    blood components. This indicates good overall health in terms of the measured parameters, though it does 
    not rule out all possible conditions.
    """
}

def create_and_save_models():
    """Create and save the initial models"""
    print("Creating initial models...")
    
    # Create a simple random forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Create synthetic training data
    # This is a simplified approach - in a real application, you would use actual patient data
    
    # Features: glucose, hemoglobin, wbc, platelets, creatinine, bun, sodium, potassium
    X = []
    y = []
    
    # Normal samples
    for _ in range(200):
        X.append([
            np.random.uniform(70, 100),  # glucose
            np.random.uniform(13, 17),   # hemoglobin
            np.random.uniform(4.5, 11),  # wbc
            np.random.uniform(150, 450), # platelets
            np.random.uniform(0.6, 1.2), # creatinine
            np.random.uniform(7, 20),    # bun
            np.random.uniform(135, 145), # sodium
            np.random.uniform(3.5, 5)    # potassium
        ])
        y.append("Normal")
    
    # Diabetes samples
    for _ in range(100):
        X.append([
            np.random.uniform(126, 300),  # High glucose
            np.random.uniform(12, 17),    # Normal/slightly low hemoglobin
            np.random.uniform(4, 12),     # Normal/slightly high wbc
            np.random.uniform(150, 450),  # Normal platelets
            np.random.uniform(0.6, 1.3),  # Normal/slightly high creatinine
            np.random.uniform(7, 23),     # Normal/slightly high bun
            np.random.uniform(133, 145),  # Normal/slightly low sodium
            np.random.uniform(3.5, 5)     # Normal potassium
        ])
        y.append("Diabetes")
    
    # Anemia samples
    for _ in range(100):
        X.append([
            np.random.uniform(70, 110),   # Normal glucose
            np.random.uniform(7, 11.5),   # Low hemoglobin
            np.random.uniform(4, 11),     # Normal wbc
            np.random.uniform(150, 500),  # Normal/high platelets
            np.random.uniform(0.6, 1.2),  # Normal creatinine
            np.random.uniform(7, 20),     # Normal bun
            np.random.uniform(135, 145),  # Normal sodium
            np.random.uniform(3.5, 5)     # Normal potassium
        ])
        y.append("Anemia")
    
    # Kidney Disease samples
    for _ in range(100):
        X.append([
            np.random.uniform(70, 110),   # Normal glucose
            np.random.uniform(10, 15),    # Normal/slightly low hemoglobin
            np.random.uniform(4, 11),     # Normal wbc
            np.random.uniform(130, 450),  # Normal/slightly low platelets
            np.random.uniform(1.3, 4),    # High creatinine
            np.random.uniform(20, 60),    # High bun
            np.random.uniform(130, 150),  # Normal/high sodium
            np.random.uniform(4, 6)       # Normal/high potassium
        ])
        y.append("Kidney Disease")
    
    # Infection samples
    for _ in range(100):
        X.append([
            np.random.uniform(70, 130),   # Normal/high glucose (stress response)
            np.random.uniform(11, 16),    # Normal/slightly low hemoglobin
            np.random.uniform(11, 20),    # High wbc
            np.random.uniform(150, 500),  # Normal/high platelets
            np.random.uniform(0.6, 1.3),  # Normal creatinine
            np.random.uniform(7, 25),     # Normal/high bun
            np.random.uniform(133, 145),  # Normal/slightly low sodium
            np.random.uniform(3.5, 5)     # Normal potassium
        ])
        y.append("Infection")
    
    X = np.array(X)
    
    # Create and fit a scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train the model
    clf.fit(X_scaled, y)
    
    # Save the model and scaler
    joblib.dump(clf, 'models/blood_analysis_rf.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    
    print("Models saved successfully!")
    return clf, scaler

def load_models():
    """Load the trained models or create if they don't exist"""
    model_path = Path('models/blood_analysis_rf.joblib')
    scaler_path = Path('models/scaler.joblib')
    
    if not model_path.exists() or not scaler_path.exists():
        clf, scaler = create_and_save_models()
    else:
        clf = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    
    return clf, scaler

def make_prediction(input_data):
    """Make a prediction using the loaded models"""
    clf, scaler = load_models()
    
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    
    # Make a prediction
    prediction = clf.predict(input_scaled)
    
    # Get prediction probabilities
    proba = clf.predict_proba(input_scaled)[0]
    classes = clf.classes_
    
    # Create a dictionary of condition probabilities
    probabilities = {cond: prob for cond, prob in zip(classes, proba)}
    
    return prediction, probabilities