#!/usr/bin/env python3
"""
Simple Glaucoma Prediction Demo
==============================
Simplified version for quick testing and demonstration

Usage: python simple_glaucoma_demo.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pickle
import os
from datetime import datetime

def collect_patient_data():
    """Collect patient information with simple validation"""
    print("\n" + "🏥 GLAUCOMA SCREENING SYSTEM" + "\n" + "="*40)
    
    # Age
    age = int(input("Enter age (18-100): "))
    
    # Gender
    gender_input = input("Enter gender (M/F): ").upper()
    gender = 1 if gender_input == 'M' else 0
    
    # Family History
    family = input("Family history of glaucoma? (Y/N): ").upper()
    family_history = 1 if family == 'Y' else 0
    
    # Medical History
    medical = input("Any medical conditions (diabetes/BP/heart)? (Y/N): ").upper()
    medical_history = 1 if medical == 'Y' else 0
    
    return {
        'Age': age,
        'Gender': gender,
        'FamilyHistory': family_history,
        'MedicalHistory': medical_history
    }

def get_average_iop():
    """Get average IOP from sensor data"""
    print("\n📊 Reading IOP sensor data...")
    
    try:
        # Read sensor data
        df = pd.read_csv('sensor_data.csv')
        
        # Get last 10 readings (or all if less than 10)
        iop_readings = df['IOP'].tail(10).tolist()
        
        # Filter valid readings (5-50 mmHg range)
        valid_readings = [iop for iop in iop_readings if 5 <= iop <= 50]
        
        if not valid_readings:
            print("❌ No valid IOP readings found!")
            return None
            
        avg_iop = np.mean(valid_readings)
        print(f"✅ Average IOP from {len(valid_readings)} readings: {avg_iop:.2f} mmHg")
        
        return avg_iop
        
    except Exception as e:
        print(f"❌ Error reading sensor data: {e}")
        return None

def train_simple_model():
    """Train a simple glaucoma prediction model"""
    print("\n🤖 Training ML model...")
    
    try:
        # Load dataset
        df = pd.read_csv('glaucoma_dataset.csv')
        
        # Add simulated additional features for demo
        np.random.seed(42)
        df['FamilyHistory'] = np.random.binomial(1, 0.3, len(df))
        df['MedicalHistory'] = np.random.binomial(1, 0.4, len(df))
        
        # Prepare features
        features = ['Age', 'Gender', 'IOP', 'FamilyHistory', 'MedicalHistory']
        X = df[features]
        y = df['Diagnosis']
        
        # Balance dataset
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        # Train model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_balanced)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y_balanced)
        
        # Save model
        with open('simple_model.pkl', 'wb') as f:
            pickle.dump((model, scaler), f)
        
        print("✅ Model trained and saved!")
        return model, scaler
        
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return None, None

def load_or_train_model():
    """Load existing model or train new one"""
    if os.path.exists('simple_model.pkl'):
        try:
            with open('simple_model.pkl', 'rb') as f:
                model, scaler = pickle.load(f)
            print("✅ Model loaded successfully!")
            return model, scaler
        except:
            pass
    
    return train_simple_model()

def predict_glaucoma(model, scaler, patient_data, iop):
    """Make glaucoma prediction"""
    try:
        # Prepare input
        features = np.array([[
            patient_data['Age'],
            patient_data['Gender'], 
            iop,
            patient_data['FamilyHistory'],
            patient_data['MedicalHistory']
        ]])
        
        # Scale and predict
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        return prediction, probability[1]  # Return prediction and glaucoma probability
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None, None

def display_results(patient_data, iop, prediction, risk_prob):
    """Display prediction results"""
    print("\n" + "="*50)
    print("🔬 GLAUCOMA SCREENING RESULTS")
    print("="*50)
    
    # Patient info
    gender_text = "Male" if patient_data['Gender'] == 1 else "Female"
    family_text = "Yes" if patient_data['FamilyHistory'] == 1 else "No"
    medical_text = "Yes" if patient_data['MedicalHistory'] == 1 else "No"
    
    print(f"👤 Patient: {patient_data['Age']} year old {gender_text}")
    print(f"📊 Average IOP: {iop:.2f} mmHg")
    print(f"👨‍👩‍👧‍👦 Family History: {family_text}")
    print(f"📋 Medical History: {medical_text}")
    
    print("\n🎯 PREDICTION:")
    
    if prediction == 1:
        print("🚨 HIGH GLAUCOMA RISK")
        print(f"⚠️ Risk Probability: {risk_prob:.1%}")
        recommendation = "URGENT: Consult ophthalmologist immediately"
    else:
        print("✅ LOW GLAUCOMA RISK")
        print(f"📈 Risk Probability: {risk_prob:.1%}")
        recommendation = "Continue regular eye checkups"
    
    print(f"\n💡 Recommendation: {recommendation}")
    
    # Risk factors
    print("\n⚠️ Risk Factors:")
    if patient_data['Age'] > 60:
        print(f"   • Advanced age ({patient_data['Age']} years)")
    if iop > 21:
        print(f"   • Elevated IOP ({iop:.1f} mmHg)")
    if patient_data['FamilyHistory'] == 1:
        print("   • Family history of glaucoma")
    if patient_data['MedicalHistory'] == 1:
        print("   • Relevant medical history")
    
    # Save results
    save_results(patient_data, iop, prediction, risk_prob)
    print("\n💾 Results saved to screening_results.csv")
    print("="*50)

def save_results(patient_data, iop, prediction, risk_prob):
    """Save results to CSV"""
    result = {
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Age': patient_data['Age'],
        'Gender': 'Male' if patient_data['Gender'] == 1 else 'Female',
        'Average_IOP': f"{iop:.2f}",
        'Family_History': 'Yes' if patient_data['FamilyHistory'] == 1 else 'No',
        'Medical_History': 'Yes' if patient_data['MedicalHistory'] == 1 else 'No',
        'Risk_Level': 'HIGH' if prediction == 1 else 'LOW',
        'Risk_Probability': f"{risk_prob:.3f}"
    }
    
    df = pd.DataFrame([result])
    if os.path.exists('screening_results.csv'):
        df.to_csv('screening_results.csv', mode='a', header=False, index=False)
    else:
        df.to_csv('screening_results.csv', index=False)

def main():
    """Main function"""
    print("🚀 Starting Simple Glaucoma Screening...")
    
    # Load/train model
    model, scaler = load_or_train_model()
    if model is None:
        print("❌ Failed to load/train model!")
        return
    
    # Collect patient data
    patient_data = collect_patient_data()
    
    # Get IOP reading
    avg_iop = get_average_iop()
    if avg_iop is None:
        print("❌ Failed to get IOP reading!")
        return
    
    # Make prediction
    print("\n🧠 Making prediction...")
    prediction, risk_prob = predict_glaucoma(model, scaler, patient_data, avg_iop)
    
    if prediction is None:
        print("❌ Prediction failed!")
        return
    
    # Display results
    display_results(patient_data, avg_iop, prediction, risk_prob)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Program interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
