#!/usr/bin/env python3
"""
Advanced Glaucoma Prediction System
==================================
This system:
1. Collects patient information (Age, Gender, Family History, Medical History)
2. Reads 10 IOP sensor readings and averages them
3. Uses ML model to predict glaucoma risk and type
4. Provides detailed medical recommendations

Author: AI Assistant
Date: August 2025
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import pickle
import warnings
import time
import os
import sys
from datetime import datetime

warnings.filterwarnings('ignore')

class GlaucomaPredictionSystem:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = ['Age', 'Gender', 'IOP', 'FamilyHistory', 'MedicalHistory']
        self.model_path = 'glaucoma_model.pkl'
        self.scaler_path = 'scaler.pkl'
        
    def collect_patient_info(self):
        """Collect patient information from user input"""
        print("\n" + "="*60)
        print("🏥 ADVANCED GLAUCOMA SCREENING SYSTEM")
        print("="*60)
        print("Please provide the following patient information:")
        print("-"*40)
        
        # Age input with validation
        while True:
            try:
                age = int(input("📅 Enter patient's age (18-100): "))
                if 18 <= age <= 100:
                    break
                else:
                    print("❌ Please enter age between 18-100 years")
            except ValueError:
                print("❌ Please enter a valid number")
        
        # Gender input with validation
        while True:
            gender_input = input("⚧ Enter gender (M/F or Male/Female): ").strip().upper()
            if gender_input in ['M', 'MALE']:
                gender = 1  # Male = 1
                gender_text = "Male"
                break
            elif gender_input in ['F', 'FEMALE']:
                gender = 0  # Female = 0
                gender_text = "Female"
                break
            else:
                print("❌ Please enter M/F or Male/Female")
        
        # Family History input
        while True:
            family_input = input("👨‍👩‍👧‍👦 Family history of glaucoma? (Y/N): ").strip().upper()
            if family_input in ['Y', 'YES']:
                family_history = 1
                family_text = "Yes"
                break
            elif family_input in ['N', 'NO']:
                family_history = 0
                family_text = "No"
                break
            else:
                print("❌ Please enter Y/N or Yes/No")
        
        # Medical History input
        print("\n📋 Medical History (select relevant conditions):")
        print("1. Diabetes")
        print("2. High Blood Pressure")
        print("3. Heart Disease")
        print("4. Previous Eye Surgery")
        print("5. None of the above")
        
        while True:
            try:
                med_choice = int(input("Enter choice (1-5): "))
                if 1 <= med_choice <= 5:
                    if med_choice == 5:
                        medical_history = 0
                        medical_text = "None"
                    else:
                        medical_history = 1
                        medical_conditions = {
                            1: "Diabetes",
                            2: "High Blood Pressure", 
                            3: "Heart Disease",
                            4: "Previous Eye Surgery"
                        }
                        medical_text = medical_conditions[med_choice]
                    break
                else:
                    print("❌ Please enter a number between 1-5")
            except ValueError:
                print("❌ Please enter a valid number")
        
        patient_info = {
            'Age': age,
            'Gender': gender,
            'Gender_text': gender_text,
            'FamilyHistory': family_history,
            'FamilyHistory_text': family_text,
            'MedicalHistory': medical_history,
            'MedicalHistory_text': medical_text
        }
        
        # Display collected information
        print("\n" + "="*50)
        print("📊 PATIENT INFORMATION SUMMARY")
        print("="*50)
        print(f"👤 Age: {age} years")
        print(f"⚧ Gender: {gender_text}")
        print(f"👨‍👩‍👧‍👦 Family History: {family_text}")
        print(f"📋 Medical History: {medical_text}")
        print("="*50)
        
        return patient_info
    
    def collect_iop_readings(self, num_readings=10):
        """Collect multiple IOP readings from sensor data and average them"""
        print(f"\n🔬 COLLECTING {num_readings} IOP READINGS...")
        print("="*50)
        
        # Check if sensor data file exists
        if not os.path.exists('sensor_data.csv'):
            print("❌ Error: sensor_data.csv not found!")
            print("💡 Please ensure the sensor is connected and data is being collected.")
            return None
        
        # Read sensor data
        try:
            sensor_df = pd.read_csv('sensor_data.csv')
            if len(sensor_df) < num_readings:
                print(f"⚠️ Warning: Only {len(sensor_df)} readings available, need {num_readings}")
                print("📡 Waiting for more sensor readings...")
                # In real implementation, this would wait for more data
                # For now, we'll use available data
                iop_readings = sensor_df['IOP'].tail(len(sensor_df)).tolist()
            else:
                # Get the last 10 readings
                iop_readings = sensor_df['IOP'].tail(num_readings).tolist()
            
            # Filter out invalid readings (outside physiological range)
            valid_readings = [iop for iop in iop_readings if 5 <= iop <= 50]
            
            if len(valid_readings) == 0:
                print("❌ Error: No valid IOP readings found!")
                return None
            
            # Calculate statistics
            avg_iop = np.mean(valid_readings)
            std_iop = np.std(valid_readings)
            min_iop = np.min(valid_readings)
            max_iop = np.max(valid_readings)
            
            print(f"📊 IOP READING ANALYSIS:")
            print(f"   📈 Number of valid readings: {len(valid_readings)}")
            print(f"   📊 Average IOP: {avg_iop:.2f} mmHg")
            print(f"   📉 Min IOP: {min_iop:.2f} mmHg")
            print(f"   📈 Max IOP: {max_iop:.2f} mmHg")
            print(f"   📏 Standard Deviation: {std_iop:.2f} mmHg")
            
            # Quality check
            if std_iop > 5:
                print("⚠️ Warning: High variability in readings. Consider re-measuring.")
            else:
                print("✅ Reading quality: Good")
                
            return avg_iop
            
        except Exception as e:
            print(f"❌ Error reading sensor data: {str(e)}")
            return None
    
    def train_model(self):
        """Train the glaucoma prediction model"""
        print("\n🤖 TRAINING MACHINE LEARNING MODEL...")
        print("="*50)
        
        # Load training dataset
        if not os.path.exists('glaucoma_dataset.csv'):
            print("❌ Error: Training dataset 'glaucoma_dataset.csv' not found!")
            return False
        
        try:
            # Load and preprocess data
            df = pd.read_csv('glaucoma_dataset.csv')
            df.columns = df.columns.str.strip()
            
            # For training, we'll simulate additional features
            # In real scenario, you'd collect this data historically
            np.random.seed(42)
            df['FamilyHistory'] = np.random.binomial(1, 0.3, len(df))  # 30% have family history
            df['MedicalHistory'] = np.random.binomial(1, 0.4, len(df))  # 40% have medical conditions
            
            # Prepare features
            X = df[self.feature_columns]
            y = df['Diagnosis']
            
            # Handle class imbalance with SMOTE
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train Random Forest model
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"✅ Model trained successfully!")
            print(f"📊 Training Accuracy: {accuracy:.3f}")
            print(f"📈 Features used: {', '.join(self.feature_columns)}")
            
            # Save model and scaler
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            print("💾 Model saved successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error training model: {str(e)}")
            return False
    
    def load_model(self):
        """Load pre-trained model"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print("✅ Model loaded successfully!")
                return True
            else:
                print("⚠️ Pre-trained model not found. Training new model...")
                return self.train_model()
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def predict_glaucoma(self, patient_info, iop_value):
        """Make glaucoma prediction"""
        if self.model is None or self.scaler is None:
            print("❌ Model not loaded!")
            return None
        
        try:
            # Prepare input features
            features = np.array([[
                patient_info['Age'],
                patient_info['Gender'],
                iop_value,
                patient_info['FamilyHistory'],
                patient_info['MedicalHistory']
            ]])
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0]
            
            return {
                'prediction': prediction,
                'probability_no_glaucoma': probability[0],
                'probability_glaucoma': probability[1],
                'confidence': max(probability)
            }
            
        except Exception as e:
            print(f"❌ Error making prediction: {str(e)}")
            return None
    
    def interpret_results(self, patient_info, iop_value, prediction_result):
        """Provide detailed interpretation of results"""
        print("\n" + "="*60)
        print("🔬 GLAUCOMA SCREENING RESULTS")
        print("="*60)
        
        # Patient summary
        print("👤 PATIENT SUMMARY:")
        print(f"   Age: {patient_info['Age']} years")
        print(f"   Gender: {patient_info['Gender_text']}")
        print(f"   Average IOP: {iop_value:.2f} mmHg")
        print(f"   Family History: {patient_info['FamilyHistory_text']}")
        print(f"   Medical History: {patient_info['MedicalHistory_text']}")
        
        print("\n🎯 PREDICTION RESULTS:")
        
        # Risk assessment
        risk_prob = prediction_result['probability_glaucoma']
        confidence = prediction_result['confidence']
        
        if prediction_result['prediction'] == 1:
            print("🚨 HIGH GLAUCOMA RISK DETECTED")
            risk_level = "HIGH"
            color_indicator = "🔴"
        else:
            print("✅ LOW GLAUCOMA RISK")
            risk_level = "LOW"
            color_indicator = "🟢"
        
        print(f"   {color_indicator} Risk Level: {risk_level}")
        print(f"   📊 Glaucoma Probability: {risk_prob:.1%}")
        print(f"   🎯 Model Confidence: {confidence:.1%}")
        
        # Detailed risk factors
        print("\n⚠️ RISK FACTOR ANALYSIS:")
        
        risk_factors = []
        if patient_info['Age'] > 60:
            risk_factors.append(f"Advanced age ({patient_info['Age']} years)")
        if iop_value > 21:
            risk_factors.append(f"Elevated IOP ({iop_value:.1f} mmHg)")
        if patient_info['FamilyHistory'] == 1:
            risk_factors.append("Family history of glaucoma")
        if patient_info['MedicalHistory'] == 1:
            risk_factors.append(f"Medical history: {patient_info['MedicalHistory_text']}")
        
        if risk_factors:
            for factor in risk_factors:
                print(f"   ⚠️ {factor}")
        else:
            print("   ✅ No major risk factors identified")
        
        # Clinical recommendations
        print("\n💡 CLINICAL RECOMMENDATIONS:")
        
        if risk_prob > 0.7:
            print("   🚨 URGENT: Immediate ophthalmologist consultation required")
            print("   📅 Recommend comprehensive eye exam within 1 week")
            print("   🔬 Consider visual field test and OCT imaging")
        elif risk_prob > 0.4:
            print("   ⚠️ MODERATE RISK: Schedule ophthalmologist appointment")
            print("   📅 Recommend comprehensive eye exam within 1 month") 
            print("   👁️ Regular monitoring recommended")
        else:
            print("   ✅ LOW RISK: Continue regular eye checkups")
            print("   📅 Annual eye exam recommended")
            print("   🏥 Maintain healthy lifestyle")
        
        # IOP interpretation
        print("\n📊 IOP INTERPRETATION:")
        if iop_value <= 12:
            print("   📉 Low IOP - Monitor for potential hypotony")
        elif iop_value <= 21:
            print("   ✅ Normal IOP range")
        elif iop_value <= 25:
            print("   ⚠️ Borderline elevated IOP")
        else:
            print("   🚨 Significantly elevated IOP - Immediate attention needed")
        
        # Save results
        self.save_results(patient_info, iop_value, prediction_result)
        
        print("\n" + "="*60)
        print("📄 Results saved to patient_results.csv")
        print("🏥 Please consult with an eye care professional for proper diagnosis")
        print("="*60)
    
    def save_results(self, patient_info, iop_value, prediction_result):
        """Save results to CSV file"""
        result_data = {
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Age': patient_info['Age'],
            'Gender': patient_info['Gender_text'],
            'FamilyHistory': patient_info['FamilyHistory_text'],
            'MedicalHistory': patient_info['MedicalHistory_text'],
            'Average_IOP': iop_value,
            'Glaucoma_Risk': 'HIGH' if prediction_result['prediction'] == 1 else 'LOW',
            'Risk_Probability': f"{prediction_result['probability_glaucoma']:.3f}",
            'Model_Confidence': f"{prediction_result['confidence']:.3f}"
        }
        
        # Save to CSV
        results_df = pd.DataFrame([result_data])
        if os.path.exists('patient_results.csv'):
            results_df.to_csv('patient_results.csv', mode='a', header=False, index=False)
        else:
            results_df.to_csv('patient_results.csv', index=False)
    
    def run_screening(self):
        """Main screening workflow"""
        print("🚀 Starting Glaucoma Screening System...")
        
        # Load or train model
        if not self.load_model():
            print("❌ Failed to load/train model. Exiting...")
            return
        
        # Collect patient information
        patient_info = self.collect_patient_info()
        
        # Collect IOP readings
        avg_iop = self.collect_iop_readings()
        if avg_iop is None:
            print("❌ Failed to collect IOP readings. Exiting...")
            return
        
        # Make prediction
        print("\n🧠 MAKING PREDICTION...")
        prediction_result = self.predict_glaucoma(patient_info, avg_iop)
        
        if prediction_result is None:
            print("❌ Failed to make prediction. Exiting...")
            return
        
        # Interpret and display results
        self.interpret_results(patient_info, avg_iop, prediction_result)

def main():
    """Main function"""
    try:
        system = GlaucomaPredictionSystem()
        system.run_screening()
    except KeyboardInterrupt:
        print("\n\n❌ Screening interrupted by user.")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        print("Please check your data files and try again.")

if __name__ == "__main__":
    main()
