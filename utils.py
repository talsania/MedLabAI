import pandas as pd
import numpy as np

def create_sample_data():
    """Create sample data for demonstration"""
    
    # Create an empty dataframe
    columns = ['glucose', 'hemoglobin', 'wbc', 'platelets', 'creatinine', 'bun', 'sodium', 'potassium', 'condition']
    sample_data = pd.DataFrame(columns=columns)
    
    # Normal sample
    sample_data.loc[0] = [92, 14.5, 7.2, 250, 0.9, 15, 140, 4.2, 'Normal']
    
    # Diabetes sample
    sample_data.loc[1] = [180, 14.0, 8.1, 245, 1.0, 17, 138, 4.5, 'Diabetes']
    
    # Anemia sample
    sample_data.loc[2] = [95, 9.5, 6.8, 320, 0.8, 14, 139, 4.0, 'Anemia']
    
    # Kidney Disease sample
    sample_data.loc[3] = [90, 13.2, 7.5, 210, 2.5, 35, 144, 5.2, 'Kidney Disease']
    
    # Infection sample
    sample_data.loc[4] = [105, 13.5, 16.2, 380, 0.9, 18, 137, 4.1, 'Infection']
    
    # Severe Diabetes
    sample_data.loc[5] = [320, 13.8, 9.5, 230, 1.3, 22, 136, 4.7, 'Diabetes']
    
    # Severe Anemia
    sample_data.loc[6] = [88, 6.5, 7.0, 390, 0.7, 16, 138, 3.9, 'Anemia']
    
    # Severe Kidney Disease
    sample_data.loc[7] = [95, 11.5, 8.2, 180, 4.8, 65, 148, 5.8, 'Kidney Disease']
    
    # Severe Infection
    sample_data.loc[8] = [115, 12.8, 25.0, 490, 1.1, 24, 135, 4.3, 'Infection']
    
    # Mixed conditions - Diabetes and Kidney Disease
    sample_data.loc[9] = [175, 12.0, 7.8, 210, 2.2, 30, 142, 5.0, 'Kidney Disease']
    
    return sample_data

def preprocess_input(input_data):
    """Preprocess input data before making predictions"""
    
    # Ensure all required columns are present
    required_columns = ['glucose', 'hemoglobin', 'wbc', 'platelets', 'creatinine', 'bun', 'sodium', 'potassium']
    
    # Check if all required columns exist
    for col in required_columns:
        if col not in input_data.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Reorder columns to match model expectations
    input_data = input_data[required_columns]
    
    # Handle missing values if any (replace with means)
    if input_data.isnull().any().any():
        # Define reasonable defaults for missing values
        defaults = {
            'glucose': 95.0,
            'hemoglobin': 14.0,
            'wbc': 7.5,
            'platelets': 250.0,
            'creatinine': 0.9,
            'bun': 15.0,
            'sodium': 140.0,
            'potassium': 4.0
        }
        
        # Fill missing values with defaults
        for col in input_data.columns:
            if input_data[col].isnull().any() and col in defaults:
                input_data[col] = input_data[col].fillna(defaults[col])
    
    return input_data