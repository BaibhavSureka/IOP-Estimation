"""
🧪 FIXED GLAUCOMA PREDICTION SYSTEM - INTERACTIVE TESTING
=========================================================
Command-line interactive testing with user input

This script provides an interactive interface for real patient data input
and glaucoma prediction using the trained models.

Usage:
1. Run training script first (google_colab_training.py)
2. Run this script: python interactive_testing.py
3. Follow the prompts to enter patient data
"""

import pandas as pd
import numpy as np
import pickle
import warnings
import os
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

print("🧪 FIXED GLAUCOMA PREDICTION SYSTEM - INTERACTIVE TESTING")
print("=" * 65)

# ==========================================
# MODEL LOADING
# ==========================================

def load_trained_models():
    """Load all trained models and scalers"""
    print("\n📂 Loading trained models...")
    
    try:
        models = {}
        model_files = [
            'iop_regressor.pkl',
            'iop_scaler.pkl', 
            'diagnosis_classifier.pkl',
            'clinical_scaler.pkl'
        ]
        
        for file in model_files:
            if os.path.exists(file):
                with open(file, 'rb') as f:
                    model_name = file.replace('.pkl', '')
                    models[model_name] = pickle.load(f)
                print(f"   ✅ Loaded: {file}")
            else:
                print(f"   ❌ Missing: {file}")
                return None
        
        print("   🎯 All models loaded successfully!")
        return models
        
    except Exception as e:
        print(f"   ❌ Error loading models: {e}")
        print("   📋 Please run the training script first!")
        return None

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def engineer_clinical_features(patient_data):
    """Engineer clinical features for prediction"""
    features = patient_data.copy()
    
    # Age groups
    if features['Age'] <= 40:
        features['Age_Group'] = 0
    elif features['Age'] <= 60:
        features['Age_Group'] = 1
    elif features['Age'] <= 80:
        features['Age_Group'] = 2
    else:
        features['Age_Group'] = 3
    
    # IOP categories
    if features['IOP'] <= 12:
        features['IOP_Category'] = 0
    elif features['IOP'] <= 21:
        features['IOP_Category'] = 1
    elif features['IOP'] <= 30:
        features['IOP_Category'] = 2
    else:
        features['IOP_Category'] = 3
    
    # Risk factors count
    risk_factors = ['Family_History', 'Diabetes', 'Hypertension', 'Myopia', 'Previous_Eye_Surgery']
    features['Risk_Factor_Count'] = sum([features[rf] for rf in risk_factors])
    
    # Interaction features
    features['Age_IOP_Interaction'] = features['Age'] * features['IOP']
    features['High_Risk_Age_IOP'] = 1 if (features['Age'] > 60 and features['IOP'] > 21) else 0
    features['Family_High_IOP'] = 1 if (features['Family_History'] == 1 and features['IOP'] > 18) else 0
    features['Hypertension_IOP'] = features['Hypertension'] * features['IOP']
    
    return features

def assess_clinical_risk(patient_data):
    """Assess clinical risk factors"""
    risk_score = 0
    risk_factors = []
    
    # Age risk
    if patient_data['Age'] > 60:
        risk_score += 3
        risk_factors.append("Age > 60")
    elif patient_data['Age'] > 40:
        risk_score += 1
        risk_factors.append("Age > 40")
    
    # IOP risk
    if patient_data['IOP'] > 21:
        risk_score += 3
        risk_factors.append("High IOP")
    elif patient_data['IOP'] > 18:
        risk_score += 1
        risk_factors.append("Elevated IOP")
    
    # Medical history
    if patient_data['Family_History']:
        risk_score += 2
        risk_factors.append("Family History")
    
    if patient_data['Diabetes']:
        risk_score += 1
        risk_factors.append("Diabetes")
    
    if patient_data['Hypertension']:
        risk_score += 1
        risk_factors.append("Hypertension")
    
    if patient_data['Myopia']:
        risk_score += 1
        risk_factors.append("Myopia")
    
    if patient_data['Previous_Eye_Surgery']:
        risk_score += 1
        risk_factors.append("Previous Eye Surgery")
    
    # Risk level
    if risk_score >= 7:
        risk_level = "VERY HIGH"
    elif risk_score >= 5:
        risk_level = "HIGH"
    elif risk_score >= 3:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"
    
    return {
        'risk_level': risk_level,
        'risk_score': risk_score,
        'risk_factors': risk_factors
    }

# ==========================================
# USER INPUT FUNCTIONS
# ==========================================

def get_patient_data():
    """Interactive function to collect patient data"""
    print("\n" + "="*60)
    print("👤 PATIENT DATA INPUT")
    print("="*60)
    
    patient_data = {}
    
    try:
        # Basic Information
        print("\n📋 Basic Information:")
        while True:
            try:
                age_input = input("Enter patient age (years): ").strip()
                age = int(age_input)
                if 1 <= age <= 120:
                    patient_data['Age'] = age
                    break
                else:
                    print("Please enter a valid age between 1 and 120")
            except ValueError:
                print("Please enter a valid number for age")
        
        while True:
            gender_input = input("Enter gender (M/F): ").upper().strip()
            if gender_input in ['M', 'F']:
                patient_data['Gender'] = 1 if gender_input == 'M' else 0
                break
            else:
                print("Please enter M for Male or F for Female")
        
        # Medical History
        print("\n🏥 Medical History (Answer Y/N):")
        
        questions = [
            ('Family_History', "Family history of glaucoma?"),
            ('Diabetes', "History of diabetes?"),
            ('Hypertension', "History of hypertension/high blood pressure?"),
            ('Myopia', "History of myopia/nearsightedness?"),
            ('Previous_Eye_Surgery', "Previous eye surgery?")
        ]
        
        for key, question in questions:
            while True:
                answer = input(f"{question} (Y/N): ").upper().strip()
                if answer in ['Y', 'N']:
                    patient_data[key] = 1 if answer == 'Y' else 0
                    break
                else:
                    print("Please enter Y or N")
        
        # IOP Measurement
        print("\n📊 IOP (Intraocular Pressure) Measurement:")
        print("1. I have sensor data file")
        print("2. I will enter IOP value manually")
        
        while True:
            iop_choice = input("Choose option (1 or 2): ").strip()
            if iop_choice in ['1', '2']:
                break
            else:
                print("Please enter 1 or 2")
        
        if iop_choice == '1':
            # Sensor data option
            sensor_file = input("Enter sensor data filename (default: sensor_data.csv): ").strip()
            if not sensor_file:
                sensor_file = 'sensor_data.csv'
            
            calculated_iop = process_sensor_data(sensor_file)
            if calculated_iop is not None:
                patient_data['IOP'] = calculated_iop
                patient_data['IOP_Source'] = 'sensor'
            else:
                print("   ⚠️ Sensor processing failed. Please enter IOP manually:")
                while True:
                    try:
                        manual_iop = float(input("Enter IOP value (mmHg): "))
                        if 5 <= manual_iop <= 50:
                            patient_data['IOP'] = manual_iop
                            patient_data['IOP_Source'] = 'manual'
                            break
                        else:
                            print("Please enter IOP between 5 and 50 mmHg")
                    except ValueError:
                        print("Please enter a valid number for IOP")
        else:
            # Manual IOP entry
            while True:
                try:
                    manual_iop = float(input("Enter IOP value (mmHg): "))
                    if 5 <= manual_iop <= 50:
                        patient_data['IOP'] = manual_iop
                        patient_data['IOP_Source'] = 'manual'
                        break
                    else:
                        print("Please enter IOP between 5 and 50 mmHg")
                except ValueError:
                    print("Please enter a valid number for IOP")
        
        print("\n✅ Patient data collected successfully!")
        return patient_data
        
    except KeyboardInterrupt:
        print("\n\n👋 Input cancelled by user")
        return None
    except Exception as e:
        print(f"\n❌ Error collecting patient data: {e}")
        return None

def process_sensor_data(sensor_file):
    """Process sensor data to calculate IOP"""
    try:
        if not os.path.exists(sensor_file):
            print(f"   ❌ Sensor file not found: {sensor_file}")
            return None
        
        print(f"\n🔧 Processing sensor data from: {sensor_file}")
        
        # Load sensor data
        sensor_data = pd.read_csv(sensor_file)
        print(f"   📊 Loaded {len(sensor_data)} sensor readings")
        
        # Get IOP column
        iop_col = None
        for col in ['IOP', 'iop', 'Intraocular_Pressure']:
            if col in sensor_data.columns:
                iop_col = col
                break
        
        if iop_col is None and len(sensor_data.columns) >= 3:
            iop_col = sensor_data.columns[2]  # Use third column
        
        if iop_col is None:
            print("   ❌ No IOP column found in sensor data")
            return None
        
        # Clean and validate IOP values
        iop_values = sensor_data[iop_col].dropna()
        valid_iop = iop_values[(iop_values > 5) & (iop_values < 50)]
        
        if len(valid_iop) == 0:
            print("   ❌ No valid IOP readings found")
            return None
        
        # Calculate robust IOP using median
        robust_iop = valid_iop.median()
        
        print(f"   ✅ Calculated IOP: {robust_iop:.2f} mmHg")
        print(f"   📊 Valid readings: {len(valid_iop)}/{len(sensor_data)}")
        
        return robust_iop
        
    except Exception as e:
        print(f"   ❌ Error processing sensor data: {e}")
        return None

# ==========================================
# PREDICTION FUNCTIONS
# ==========================================

def make_prediction(patient_data, models):
    """Make glaucoma prediction using trained models"""
    try:
        print("\n🤖 Making AI Prediction...")
        
        # Engineer features
        engineered_features = engineer_clinical_features(patient_data)
        
        # Prepare feature vector
        feature_cols = ['Age', 'Gender', 'Family_History', 'Diabetes', 'Hypertension', 
                       'Myopia', 'Previous_Eye_Surgery', 'IOP', 'Age_Group', 'IOP_Category',
                       'Risk_Factor_Count', 'Age_IOP_Interaction', 'High_Risk_Age_IOP',
                       'Family_High_IOP', 'Hypertension_IOP']
        
        feature_vector = np.array([[engineered_features[col] for col in feature_cols]])
        
        # Scale features
        feature_vector_scaled = models['clinical_scaler'].transform(feature_vector)
        
        # Make prediction
        diagnosis_prob = models['diagnosis_classifier'].predict_proba(feature_vector_scaled)[0]
        diagnosis_pred = models['diagnosis_classifier'].predict(feature_vector_scaled)[0]
        
        # Get probability of glaucoma
        classes = models['diagnosis_classifier'].classes_
        if 'Glaucoma' in classes:
            glaucoma_idx = list(classes).index('Glaucoma')
            glaucoma_probability = diagnosis_prob[glaucoma_idx]
        else:
            # Handle numeric labels
            glaucoma_probability = diagnosis_prob[1] if len(diagnosis_prob) > 1 else diagnosis_prob[0]
        
        # Assess clinical risk
        risk_assessment = assess_clinical_risk(patient_data)
        
        results = {
            'patient_summary': {
                'age': patient_data['Age'],
                'gender': 'Male' if patient_data['Gender'] == 1 else 'Female',
                'iop': patient_data['IOP'],
                'iop_source': patient_data.get('IOP_Source', 'unknown')
            },
            'predictions': {
                'diagnosis': diagnosis_pred,
                'glaucoma_probability': glaucoma_probability,
                'model_confidence': max(diagnosis_prob)
            },
            'risk_assessment': risk_assessment
        }
        
        print("   ✅ Prediction completed successfully!")
        return results
        
    except Exception as e:
        print(f"   ❌ Prediction error: {e}")
        return None

def display_results(results):
    """Display prediction results"""
    if results is None:
        print("❌ No results to display")
        return
    
    print("\n" + "="*70)
    print("📋 GLAUCOMA PREDICTION RESULTS")
    print("="*70)
    
    # Patient Profile
    patient = results['patient_summary']
    print(f"\n👤 Patient Profile:")
    print(f"   Age: {patient['age']} years")
    print(f"   Gender: {patient['gender']}")
    print(f"   IOP: {patient['iop']:.1f} mmHg")
    print(f"   📋 IOP source: {patient['iop_source'].title()}")
    
    # Risk Assessment
    risk = results['risk_assessment']
    print(f"\n⚠️ Clinical Risk Assessment:")
    print(f"   Risk Level: {risk['risk_level']}")
    print(f"   Risk Score: {risk['risk_score']}/10")
    if risk['risk_factors']:
        print(f"   Risk Factors: {', '.join(risk['risk_factors'])}")
    else:
        print(f"   Risk Factors: None identified")
    
    # AI Predictions
    pred = results['predictions']
    print(f"\n🎯 AI Prediction Results:")
    print(f"   Glaucoma Probability: {pred['glaucoma_probability']:.1%}")
    print(f"   Diagnosis: {pred['diagnosis']}")
    print(f"   Model Confidence: {pred['model_confidence']:.1%}")
    
    # Clinical Recommendations
    print(f"\n💡 Clinical Recommendations:")
    
    glaucoma_prob = pred['glaucoma_probability']
    risk_level = risk['risk_level']
    
    if glaucoma_prob >= 0.7 or risk_level == "VERY HIGH":
        print("   🚨 HIGH RISK: Immediate ophthalmological evaluation recommended")
        print("   📋 Consider: Comprehensive eye exam, visual field testing, OCT")
        print("   ⏰ Urgency: Schedule appointment within 1-2 weeks")
    elif glaucoma_prob >= 0.5 or risk_level in ["HIGH", "MODERATE"]:
        print("   ⚠️ MODERATE RISK: Follow-up with eye care professional")
        print("   📋 Consider: Regular monitoring, repeat IOP measurement")
        print("   ⏰ Urgency: Schedule appointment within 1 month")
    else:
        print("   ✅ LOW RISK: Routine eye care recommended")
        print("   📋 Consider: Annual eye exams, maintain healthy lifestyle")
        print("   ⏰ Urgency: Regular check-ups as per age guidelines")
    
    if patient['iop'] > 21:
        print("   📈 High IOP detected: Monitor closely for pressure control")
    
    if risk['risk_level'] in ['HIGH', 'VERY HIGH']:
        print("   👥 Multiple risk factors present: Enhanced monitoring recommended")

# ==========================================
# MAIN FUNCTION
# ==========================================

def main():
    """Main interactive testing function"""
    print("🎯 Welcome to the Interactive Glaucoma Prediction System!")
    
    # Load models
    models = load_trained_models()
    if models is None:
        print("❌ Cannot proceed without trained models.")
        return
    
    while True:
        print("\n" + "="*60)
        print("🔬 INTERACTIVE GLAUCOMA PREDICTION MENU")
        print("="*60)
        print("1. 👤 Enter new patient data for prediction")
        print("2. 🧪 Run quick test with sample data")
        print("3. 📊 View model information")
        print("4. 🚪 Exit")
        
        try:
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == '1':
                # Manual patient input
                print("\n🚀 Starting patient data collection...")
                patient_data = get_patient_data()
                
                if patient_data:
                    results = make_prediction(patient_data, models)
                    display_results(results)
                    
                    # Ask if user wants to save results
                    save_choice = input("\nSave results to file? (Y/N): ").upper().strip()
                    if save_choice == 'Y':
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"prediction_results_{timestamp}.txt"
                        try:
                            with open(filename, 'w') as f:
                                f.write(f"Glaucoma Prediction Results\n")
                                f.write(f"Generated: {datetime.now()}\n")
                                f.write(f"Patient Age: {results['patient_summary']['age']}\n")
                                f.write(f"Gender: {results['patient_summary']['gender']}\n")
                                f.write(f"IOP: {results['patient_summary']['iop']:.1f} mmHg\n")
                                f.write(f"Diagnosis: {results['predictions']['diagnosis']}\n")
                                f.write(f"Probability: {results['predictions']['glaucoma_probability']:.1%}\n")
                                f.write(f"Risk Level: {results['risk_assessment']['risk_level']}\n")
                            print(f"✅ Results saved to {filename}")
                        except Exception as e:
                            print(f"❌ Error saving file: {e}")
            
            elif choice == '2':
                # Quick test with sample data
                print("\n🧪 Running quick test with sample patient...")
                sample_patient = {
                    'Age': 55, 'Gender': 0, 'Family_History': 1, 'Diabetes': 0,
                    'Hypertension': 1, 'Myopia': 0, 'Previous_Eye_Surgery': 0, 
                    'IOP': 19.5, 'IOP_Source': 'test'
                }
                
                print("   Sample patient: 55-year-old female with family history and hypertension")
                results = make_prediction(sample_patient, models)
                display_results(results)
            
            elif choice == '3':
                # Model information
                print("\n📊 Model Information:")
                for model_name, model in models.items():
                    if model is not None:
                        print(f"   ✅ {model_name}: {type(model).__name__}")
                    else:
                        print(f"   ❌ {model_name}: Not loaded")
                
                print(f"\n⚠️ Current Model Status:")
                print(f"   The model accuracy is approximately 50-60%")
                print(f"   This suggests the model needs improvement")
                print(f"   Use results as preliminary screening only")
            
            elif choice == '4':
                print("\n👋 Thank you for using the Glaucoma Prediction System!")
                break
            
            else:
                print("❌ Invalid option. Please select 1-4.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

# ==========================================
# RUN MAIN FUNCTION
# ==========================================

if __name__ == "__main__":
    main()
    print("\n🎯 Interactive testing completed!")
