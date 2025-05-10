# MedLabAI

Blood test analysis system that predicts health conditions using machine learning.

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.22.0%2B-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2%2B-orange)

## Overview

MedLabAI analyzes blood test parameters to predict potential health conditions including diabetes, anemia, kidney disease, and infections. The system runs entirely offline with no external API dependencies.

## Features

- Predicts health conditions from blood test parameters
- Real-time analysis of abnormal values
- Probability visualization for each condition
- Interactive UI with parameter input controls
- Works offline - no internet required after setup

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/MedLabAI.git
cd MedLabAI

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

## Usage

1. Navigate to "Input Data" tab
2. Enter blood test parameters
3. Click "Analyze Blood Test"
4. Review prediction results and recommendations

## Conditions Detected

- **Diabetes**: Elevated glucose, potential electrolyte imbalances
- **Anemia**: Low hemoglobin, altered RBC parameters
- **Kidney Disease**: Elevated creatinine and BUN, electrolyte imbalances
- **Infection**: Elevated WBC count, possible inflammatory markers

## Input Parameters

| Parameter | Normal Range | Unit |
|-----------|--------------|------|
| Glucose | 70-100 | mg/dL |
| Hemoglobin | 12-17 | g/dL |
| WBC Count | 4.5-11.0 | ×10^9/L |
| Platelet Count | 150-450 | ×10^9/L |
| Creatinine | 0.6-1.2 | mg/dL |
| BUN | 7-20 | mg/dL |
| Sodium | 135-145 | mEq/L |
| Potassium | 3.5-5.0 | mEq/L |

## Project Structure

```
MedLabAI/
├── app.py          # Main Streamlit application
├── models.py       # ML model functions
├── utils.py        # Utility functions
└── requirements.txt
```

## Technologies

- Python 3.8+
- Streamlit
- scikit-learn
- pandas, numpy
- matplotlib, seaborn

## Limitations

- Training data is synthetic - real world accuracy will vary
- Limited to predicting common conditions only
- Not a substitute for professional medical diagnosis

## License

MIT

> Disclaimer: This application is for educational purposes only and should not be used for actual medical diagnosis. Always consult healthcare professionals for medical advice.