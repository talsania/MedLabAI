import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from models import load_models, make_prediction, condition_descriptions
from utils import create_sample_data, preprocess_input

st.set_page_config(page_title="Blood Analysis System", layout="wide")

# Initialize session state variables if not exists
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

if 'sample_data' not in st.session_state:
    st.session_state.sample_data = create_sample_data()

# App header
st.title("Medical Lab Blood Analysis System")
st.markdown("""
This application helps medical professionals analyze blood test results to identify potential conditions
such as diabetes, anemia, kidney disease, and infections. Upload lab data or enter values manually.
""")

# Check and load models
if not st.session_state.models_loaded:
    with st.spinner("Loading models (first run only)..."):
        load_models()
        st.session_state.models_loaded = True
        st.success("Models loaded successfully!")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Input Data", "Sample Data", "About"])

if page == "Input Data":
    st.header("Blood Test Parameters")
    st.info("Enter blood test values to analyze potential conditions.")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        glucose = st.number_input("Glucose (mg/dL)", 50.0, 500.0, 95.0)
        hemoglobin = st.number_input("Hemoglobin (g/dL)", 5.0, 25.0, 14.0)
        wbc = st.number_input("White Blood Cell Count (×10^9/L)", 1.0, 50.0, 7.5)
        plt_count = st.number_input("Platelet Count (×10^9/L)", 20.0, 1000.0, 250.0)
        
    with col2:
        creatinine = st.number_input("Creatinine (mg/dL)", 0.2, 15.0, 0.9)
        bun = st.number_input("Blood Urea Nitrogen (mg/dL)", 5.0, 100.0, 15.0)
        sodium = st.number_input("Sodium (mEq/L)", 120.0, 160.0, 140.0)
        potassium = st.number_input("Potassium (mEq/L)", 2.0, 8.0, 4.0)
    
    # Create a dataframe from the input
    input_data = pd.DataFrame({
        'glucose': [glucose],
        'hemoglobin': [hemoglobin],
        'wbc': [wbc],
        'platelets': [plt_count],
        'creatinine': [creatinine],
        'bun': [bun],
        'sodium': [sodium],
        'potassium': [potassium]
    })
    
    # Add a prediction button
    if st.button("Analyze Blood Test"):
        with st.spinner("Analyzing blood parameters..."):
            # Preprocess the input
            processed_input = preprocess_input(input_data)
            
            # Get predictions
            predictions, probabilities = make_prediction(processed_input)
            
            # Display results
            st.header("Analysis Results")
            
            # Create visualization of probabilities
            fig, ax = plt.subplots(figsize=(10, 5))
            conditions = list(probabilities.keys())
            probs = list(probabilities.values())
            
            # Sort by probability
            sorted_indices = np.argsort(probs)
            sorted_conditions = [conditions[i] for i in sorted_indices]
            sorted_probs = [probs[i] for i in sorted_indices]
            
            # Use a color gradient
            colors = sns.color_palette("YlOrRd", len(sorted_conditions))
            
            bars = ax.barh(sorted_conditions, sorted_probs, color=colors)
            ax.set_xlabel('Probability')
            ax.set_title('Condition Probability Analysis')
            ax.set_xlim(0, 1)
            
            # Add percentage labels
            for bar in bars:
                width = bar.get_width()
                label_x_pos = width + 0.01
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1%}',
                       va='center')
            
            st.pyplot(fig)
            
            # Display the primary prediction
            st.subheader(f"Primary Assessment: {predictions[0]}")
            
            # Display condition information
            if predictions[0] in condition_descriptions:
                st.markdown(f"**About this condition:**")
                st.markdown(condition_descriptions[predictions[0]])
            
            # Display abnormal values
            st.subheader("Parameter Analysis")
            
            reference_ranges = {
                'glucose': (70, 100, "mg/dL", "Fasting Glucose"),
                'hemoglobin': (12, 17, "g/dL", "Hemoglobin"),
                'wbc': (4.5, 11.0, "×10^9/L", "White Blood Cell Count"),
                'platelets': (150, 450, "×10^9/L", "Platelet Count"),
                'creatinine': (0.6, 1.2, "mg/dL", "Creatinine"),
                'bun': (7, 20, "mg/dL", "Blood Urea Nitrogen"),
                'sodium': (135, 145, "mEq/L", "Sodium"),
                'potassium': (3.5, 5.0, "mEq/L", "Potassium")
            }
            
            abnormal_values = []
            
            for col, value in input_data.iloc[0].items():
                if col in reference_ranges:
                    min_val, max_val, unit, name = reference_ranges[col]
                    status = "Normal"
                    
                    if value < min_val:
                        status = "Low"
                        abnormal_values.append((name, value, f"{min_val}-{max_val}", unit, status))
                    elif value > max_val:
                        status = "High"
                        abnormal_values.append((name, value, f"{min_val}-{max_val}", unit, status))
            
            if abnormal_values:
                st.markdown("**Abnormal Values:**")
                abnormal_df = pd.DataFrame(abnormal_values, 
                                         columns=["Parameter", "Value", "Reference Range", "Unit", "Status"])
                
                # Style the DataFrame with colors
                def highlight_status(s):
                    return ['background-color: #ffcccc' if x == 'High' else 
                           'background-color: #cce5ff' if x == 'Low' else '' 
                           for x in s]
                
                styled_df = abnormal_df.style.apply(highlight_status, subset=['Status'])
                st.dataframe(styled_df)
            else:
                st.success("All parameters are within normal reference ranges.")
            
            # Show recommendations
            st.subheader("Recommendations")
            if predictions[0] == "Diabetes":
                st.markdown("""
                - Consider further testing such as HbA1c for diabetes confirmation
                - Blood glucose monitoring may be needed
                - Consult with an endocrinologist
                """)
            elif predictions[0] == "Anemia":
                st.markdown("""
                - Iron studies may be needed to determine the type of anemia
                - Consider B12 and folate levels
                - Possible hematology referral depending on severity
                """)
            elif predictions[0] == "Kidney Disease":
                st.markdown("""
                - Urinalysis recommended
                - Consider kidney ultrasound
                - Nephrology consultation may be needed
                """)
            elif predictions[0] == "Infection":
                st.markdown("""
                - Consider blood cultures
                - Possible CRP or procalcitonin testing
                - Monitor vital signs
                """)
            elif predictions[0] == "Normal":
                st.markdown("""
                - Continue routine health maintenance
                - Follow standard preventive care guidelines
                """)
            
            st.warning("This system provides decision support only. All results should be interpreted by qualified healthcare professionals.")
            
elif page == "Sample Data":
    st.header("Sample Patient Data")
    st.info("This section shows sample blood test data for different conditions. You can use this data to test the system.")
    
    # Display sample data
    st.dataframe(st.session_state.sample_data)
    
    # Option to use sample data
    st.markdown("### Use Sample Data for Analysis")
    selected_index = st.selectbox("Select a sample case to analyze:", 
                                range(len(st.session_state.sample_data)),
                                format_func=lambda i: f"Patient {i+1} ({st.session_state.sample_data.iloc[i]['condition']})")
    
    if st.button("Use Selected Sample"):
        # Get the selected sample
        sample = st.session_state.sample_data.iloc[selected_index]
        
        # Create input data excluding the condition column
        input_data = pd.DataFrame({col: [val] for col, val in sample.items() if col != 'condition'})
        
        with st.spinner("Analyzing sample data..."):
            # Preprocess the input
            processed_input = preprocess_input(input_data)
            
            # Get predictions
            predictions, probabilities = make_prediction(processed_input)
            
            # Display results
            st.header("Analysis Results")
            
            # Create visualization of probabilities
            fig, ax = plt.subplots(figsize=(10, 5))
            conditions = list(probabilities.keys())
            probs = list(probabilities.values())
            
            # Sort by probability
            sorted_indices = np.argsort(probs)
            sorted_conditions = [conditions[i] for i in sorted_indices]
            sorted_probs = [probs[i] for i in sorted_indices]
            
            # Use a color gradient
            colors = sns.color_palette("YlOrRd", len(sorted_conditions))
            
            bars = ax.barh(sorted_conditions, sorted_probs, color=colors)
            ax.set_xlabel('Probability')
            ax.set_title('Condition Probability Analysis')
            ax.set_xlim(0, 1)
            
            # Add percentage labels
            for bar in bars:
                width = bar.get_width()
                label_x_pos = width + 0.01
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1%}',
                       va='center')
            
            st.pyplot(fig)
            
            # Display comparison
            st.subheader("Prediction vs. Actual")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Condition", predictions[0])
            with col2:
                st.metric("Actual Condition", sample['condition'])
                
            # Display accuracy
            if predictions[0] == sample['condition']:
                st.success("✓ Prediction matches the known condition!")
            else:
                st.error("✗ Prediction differs from the known condition.")
                st.markdown(f"The system predicted **{predictions[0]}** but the sample data is labeled as **{sample['condition']}**.")
                
            # Display condition information
            if predictions[0] in condition_descriptions:
                st.markdown(f"**About the predicted condition:**")
                st.markdown(condition_descriptions[predictions[0]])

elif page == "About":
    st.header("About This System")
    st.markdown("""
    ## Blood Analysis System

    This application is designed to help analyze blood test results and identify potential medical conditions. It uses machine learning models trained on blood test data to make predictions about possible conditions based on input parameters.

    ### Features:
    - Input blood test parameters manually
    - Use sample data for testing
    - Visualize prediction probabilities
    - Get information about identified conditions
    - See abnormal parameter highlights

    ### Conditions Currently Detected:
    - Diabetes
    - Anemia
    - Kidney Disease
    - Infection
    - Normal (no significant findings)

    ### How It Works:
    The system uses a combination of machine learning models:
    1. A scikit-learn ensemble model for initial classification
    2. A transformer-based model for more complex analysis when needed

    ### Limitations:
    - This is a decision support tool only and should not replace clinical judgment
    - The system has been trained on limited data and may not cover all conditions
    - Always consult with healthcare professionals for diagnosis and treatment

    ### Development:
    This application was built using:
    - Python
    - Streamlit for the web interface
    - scikit-learn for machine learning
    - Hugging Face transformers for advanced analysis
    - Matplotlib and Seaborn for visualization
    """)
    
    # Show parameter reference ranges
    st.subheader("Reference Ranges")
    reference_data = [
        ["Glucose (fasting)", "70-100 mg/dL"],
        ["Hemoglobin (male)", "13.5-17.5 g/dL"],
        ["Hemoglobin (female)", "12.0-15.5 g/dL"],
        ["White Blood Cell Count", "4.5-11.0 ×10^9/L"],
        ["Platelet Count", "150-450 ×10^9/L"],
        ["Creatinine", "0.6-1.2 mg/dL"],
        ["Blood Urea Nitrogen", "7-20 mg/dL"],
        ["Sodium", "135-145 mEq/L"],
        ["Potassium", "3.5-5.0 mEq/L"]
    ]
    
    st.table(pd.DataFrame(reference_data, columns=["Parameter", "Reference Range"]))

# Footer
st.markdown("---")
st.markdown("**Disclaimer**: This application is for educational and decision support purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.")