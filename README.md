# 🔬 Advanced Glaucoma Prediction System

A comprehensive machine learning system for predicting glaucoma diagnosis using sensor-derived IOP (Intraocular Pressure) measurements and clinical features. This project implements multiple ML approaches from basic models to advanced ensemble methods.

## 🎯 Project Overview

This system provides a complete pipeline for glaucoma prediction with three different implementation levels:

1. **Basic Training & Testing** - Standard ML approach
2. **Enhanced Training** - Improved feature engineering and validation
3. **Advanced Ensemble** - Multi-model ensemble with sophisticated algorithms

## 📁 Project Structure

```
IOP-Estimation-Clean/
├── 📄 ml_model.py                    # Advanced ensemble system
├── 📄 model_training.py              
├── 📄 model_testing.py               
├── 📄 server.py                      # Basic sensor data collection
├── 📊 glaucoma_dataset(1).csv        # Clinical dataset (7,453 samples)
├── 📊 sensor_data.csv                # Sensor calibration data
├── 🤖 *.pkl                          # Trained model files
├── 📁 esp/                           # Arduino/ESP32 hardware code
    ├── arduino.ino
    ├── esp.ino
    └── sensor_reading.ino
```

## 📊 Dataset Information

- **Clinical Data**: 7,453 patient records with demographic and clinical features
- **Sensor Data**: Calibration readings for IOP conversion
- **Features**: Age, Gender, IOP, Family History, Medical History, Risk Factors
- **Target**: Glaucoma diagnosis (Normal/Glaucoma)

## 🎯 Model Performance

| Model Type | Accuracy Range | Best Features |
|------------|----------------|---------------|
| Basic Models | 50-70% | Simple clinical features |
| Enhanced Models | 65-85% | Advanced feature engineering |
| Ensemble Models | 75-90% | Multi-algorithm combination |

## 🔧 System Features

### 🧠 Multiple ML Approaches
- **Basic**: Random Forest, Gradient Boosting
- **Enhanced**: Feature engineering, cross-validation
- **Advanced**: Ensemble methods (Voting, Stacking, Bagging)

### 🔍 Advanced Algorithms
- XGBoost, LightGBM, CatBoost (if available)
- Neural Networks (MLP)
- Support Vector Machines
- Naive Bayes, K-Nearest Neighbors
- Stacking and Voting ensembles

### 📈 Feature Engineering
- Age-based risk categorization
- IOP risk level classification
- Medical history risk scoring
- Statistical interactions
- Advanced clinical feature creation

### 🎮 Interactive Testing
- Command-line patient data input
- Step-by-step guidance
- Real-time prediction results
- Confidence score reporting

### 🔬 Hardware Integration
- ESP32/Arduino sensor support
- Real-time IOP measurement
- Serial communication protocols

## 🛠️ Installation & Requirements

### Core Requirements
```bash
pip install pandas numpy scikit-learn matplotlib seaborn flask
```

### Advanced Features (Optional)
```bash
pip install xgboost lightgbm catboost imbalanced-learn
```

## 🚀 Quick Start Guide

### Basic Training (Recommended for beginners)
```bash
# Train basic models
python ml_model.py
```

## 📖 Detailed Usage

### 1. Training Models

#### Basic Training
```bash
python ml_model.py
# Generates: iop_model.pkl, glaucoma_model.pkl, scaler.pkl
```

### 2. Testing & Prediction

#### API Testing
```bash
# Start the integrated server
python model_testing.py

# Send sensor data via POST to /iop
# Get predictions via POST to /predict
```

### 3. Model Files Generated

| File | Description |
|------|-------------|
| `iop_model.pkl` | Converts sensor readings to IOP values |
| `glaucoma_model.pkl` | Predicts glaucoma from clinical features |
| `scaler.pkl` | Scales input data |
| `glaucoma_ensemble_model.pkl` | Advanced ensemble model |
| `model_evaluation.png` | Performance visualization |

## 🎨 Model Architecture

### Basic Pipeline
```
Sensor Data → IOP Regression → Clinical Features → Glaucoma Classification
     ↓              ↓                  ↓                    ↓
[Scaler] → [Random Forest] → [Feature Engineering] → [Gradient Boosting]
```

### Advanced Ensemble Pipeline
```
Multiple Algorithms → Voting/Stacking → Final Prediction
        ↓ ↓ ↓
[XGB, LGB, RF, SVM] → [Meta-learner] → [Ensemble Output]
```

## 📈 Evaluation & Visualization

The system provides comprehensive evaluation:
- **Accuracy Metrics**: Precision, Recall, F1-score, AUC-ROC
- **Visualizations**: Confusion matrices, ROC curves, PR curves
- **Cross-validation**: K-fold stratified validation
- **Feature Importance**: Analysis of most predictive features

## ⚠️ Important Notes

### Medical Disclaimer
- For research and educational purposes only
- Not for clinical decision-making
- Always consult qualified healthcare professionals
- Validate with clinical expertise before any medical use

### Performance Considerations
- Model accuracy depends on training data quality
- Results may vary with different patient populations
- Regular model retraining recommended with new data
- Hardware sensor calibration affects IOP accuracy

### Data Privacy
- Ensure HIPAA compliance for medical data
- Implement proper data encryption and access controls
- Follow institutional review board (IRB) guidelines
- Maintain patient anonymization protocols

## 🤝 Contributing

- **Data Quality**: Ensure medical data privacy and ethics compliance
- **Model Validation**: Test with diverse patient populations
- **Clinical Review**: Validate changes with medical professionals
- **Documentation**: Update README for any new features
- **Testing**: Thoroughly test all modifications

## 📚 References & Citations

- Clinical glaucoma risk factors and thresholds
- IOP measurement standards and calibration
- Machine learning best practices for medical data
- Ensemble methods for healthcare applications

## 🔮 Future Enhancements

- Real-time web dashboard
- Mobile app integration
- DICOM image analysis integration
- Cloud deployment options
- Multi-language support
- Advanced deep learning models

## 📄 License

This project is for educational and research purposes. Medical applications require:
- Proper clinical validation
- Regulatory approval (FDA, CE marking, etc.)
- Institutional review board approval
- Compliance with medical device regulations

## 📩 Contact

📧 **Baibhav Sureka** - [GitHub](https://github.com/BaibhavSureka) | [LinkedIn](https://linkedin.com/in/baibhavsureka)

---

🏥 **Medical Research Disclaimer**: This system is designed for research and educational purposes only. It should not be used for clinical diagnosis or treatment decisions without proper validation and approval from qualified medical professionals and regulatory authorities.