"""
🚀 ADVANCED MULTI-MODEL ENSEMBLE GLAUCOMA PREDICTION SYSTEM
=========================================================
Fixed version addressing low accuracy issues
"""
# ==========================================
# IMPORTS AND SETUP
# ==========================================
import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesClassifier, BaggingClassifier, StackingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE, mutual_info_classif
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Try to import advanced models (install if available)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("📝 Note: XGBoost not available. Install with: pip install xgboost")
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("📝 Note: LightGBM not available. Install with: pip install lightgbm")
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("📝 Note: CatBoost not available. Install with: pip install catboost")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
print("🚀 ADVANCED MULTI-MODEL ENSEMBLE GLAUCOMA PREDICTION SYSTEM")
print("=" * 70)

# ==========================================
# FIXED DATA LOADING AND PREPROCESSING
# ==========================================
def load_and_preprocess_data():
    """Fixed data loading with proper preprocessing"""
    print("\n📊 Loading and preprocessing datasets...")
    
    try:
        # Load original dataset (not cleaned version)
        original_data = pd.read_csv('glaucoma_dataset.csv')
        print(f"   ✅ Original data loaded: {original_data.shape}")
        
        # Load sensor data
        sensor_data = pd.read_csv('sensor_data.csv')
        print(f"   ✅ Sensor data loaded: {sensor_data.shape}")
        
        # Process the original data properly
        clinical_data = process_original_data(original_data)
        print(f"   ✅ Processed clinical data: {clinical_data.shape}")
        
        # Handle sensor data
        sensor_data = preprocess_sensor_data(sensor_data)
        print(f"   📈 Processed sensor data: {sensor_data.shape}")
        
        return clinical_data, sensor_data
        
    except Exception as e:
        print(f"   ❌ Error loading data: {e}")
        return None, None

def process_original_data(df):
    """Process original dataset with proper feature extraction and diagnosis correction"""
    print("   🔬 Processing original dataset with proper feature extraction...")
    
    # Create a copy
    data = df.copy()
    
    # CRITICAL FIX: Correct the Diagnosis column
    # In the original data, "No Glaucoma" should be 0, and any glaucoma should be 1
    data['Diagnosis'] = data['Diagnosis'].apply(lambda x: 0 if x == 'No Glaucoma' else 1)
    
    # Let's verify the diagnosis distribution
    print(f"   📋 Diagnosis distribution after correction: {data['Diagnosis'].value_counts().to_dict()}")
    
    # Extract features from complex columns
    
    # 1. Visual Acuity Measurements
    def extract_visual_acuity(va):
        if pd.isna(va):
            return 0.2  # Default moderate impairment
        if isinstance(va, str):
            if 'LogMAR' in va:
                # Extract LogMAR value
                match = re.search(r'LogMAR\s+(\d+\.\d+)', va)
                if match:
                    return float(match.group(1))
            elif '/' in va:
                # Handle 20/XX format
                parts = va.split('/')
                if len(parts) == 2 and parts[0].strip() == '20':
                    try:
                        denominator = float(parts[1])
                        return 20 / denominator  # Convert to decimal
                    except:
                        return 0.2
        return 0.2  # Default
    
    data['Visual_Acuity'] = data['Visual Acuity Measurements'].apply(extract_visual_acuity)
    
    # 2. Intraocular Pressure (IOP) - already numeric
    data['IOP'] = data['Intraocular Pressure (IOP)']
    
    # 3. Cup-to-Disc Ratio (CDR) - already numeric
    data['CDR'] = data['Cup-to-Disc Ratio (CDR)']
    
    # 4. Family History
    data['Family_History'] = data['Family History'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # 5. Medical History - extract specific conditions
    def extract_medical_conditions(med_history):
        if pd.isna(med_history):
            return 0
        conditions = str(med_history).lower()
        # Check for key conditions related to glaucoma
        if 'diabetes' in conditions or 'hypertension' in conditions:
            return 2  # High risk
        elif any(cond in conditions for cond in ['heart', 'migraine', 'myopia']):
            return 1  # Moderate risk
        return 0  # Low risk
    
    data['Medical_Risk'] = data['Medical History'].apply(extract_medical_conditions)
    
    # 6. Cataract Status
    data['Cataract'] = data['Cataract Status'].apply(lambda x: 1 if x == 'Present' else 0)
    
    # 7. Angle Closure Status
    data['Angle_Closure'] = data['Angle Closure Status'].apply(lambda x: 1 if x == 'Closed' else 0)
    
    # 8. Visual Field Test Results - extract sensitivity and specificity
    def extract_visual_field(vf_result):
        if pd.isna(vf_result):
            return 0.5, 0.5  # Default moderate values
        result = str(vf_result).lower()
        sensitivity = 0.5
        specificity = 0.5
        
        # Extract sensitivity
        sens_match = re.search(r'sensitivity:\s*(\d+\.\d+)', result)
        if sens_match:
            sensitivity = float(sens_match.group(1))
        
        # Extract specificity
        spec_match = re.search(r'specificity:\s*(\d+\.\d+)', result)
        if spec_match:
            specificity = float(spec_match.group(1))
        
        return sensitivity, specificity
    
    vf_data = data['Visual Field Test Results'].apply(extract_visual_field)
    data['VF_Sensitivity'] = [x[0] for x in vf_data]
    data['VF_Specificity'] = [x[1] for x in vf_data]
    
    # 9. Optical Coherence Tomography (OCT) Results - extract RNFL and GCC thickness
    def extract_oct(oct_result):
        if pd.isna(oct_result):
            return 85, 65  # Default moderate values
        result = str(oct_result).lower()
        rnfl = 85  # Default
        gcc = 65   # Default
        
        # Extract RNFL thickness
        rnfl_match = re.search(r'rnfl thickness:\s*(\d+\.\d+)\s*µm', result)
        if rnfl_match:
            rnfl = float(rnfl_match.group(1))
        
        # Extract GCC thickness
        gcc_match = re.search(r'gcc thickness:\s*(\d+\.\d+)\s*µm', result)
        if gcc_match:
            gcc = float(gcc_match.group(1))
        
        return rnfl, gcc
    
    oct_data = data['Optical Coherence Tomography (OCT) Results'].apply(extract_oct)
    data['RNFL_Thickness'] = [x[0] for x in oct_data]
    data['GCC_Thickness'] = [x[1] for x in oct_data]
    
    # 10. Pachymetry - corneal thickness
    data['Corneal_Thickness'] = data['Pachymetry']
    
    # 11. Visual Symptoms - count symptoms
    def count_symptoms(symptoms):
        if pd.isna(symptoms):
            return 0
        symptom_list = str(symptoms).lower().split(',')
        # Count distinct symptoms
        distinct_symptoms = set()
        for symptom in symptom_list:
            symptom = symptom.strip()
            if symptom:
                distinct_symptoms.add(symptom)
        return len(distinct_symptoms)
    
    data['Symptom_Count'] = data['Visual Symptoms'].apply(count_symptoms)
    
    # 12. Glaucoma Type - extract type information
    def encode_glaucoma_type(glaucoma_type):
        if pd.isna(glaucoma_type) or glaucoma_type == 'No Glaucoma':
            return 0
        glaucoma_type = str(glaucoma_type).lower()
        if 'primary open-angle' in glaucoma_type:
            return 1
        elif 'angle-closure' in glaucoma_type or 'angle closure' in glaucoma_type:
            return 2
        elif 'normal tension' in glaucoma_type:
            return 3
        elif 'congenital' in glaucoma_type:
            return 4
        elif 'juvenile' in glaucoma_type:
            return 5
        else:
            return 6  # Other types
    
    data['Glaucoma_Type'] = data['Glaucoma Type'].apply(encode_glaucoma_type)
    
    # 13. Gender
    data['Gender'] = data['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
    
    # 14. Age - already numeric
    
    # Select relevant columns
    relevant_columns = [
        'Age', 'Gender', 'Visual_Acuity', 'IOP', 'CDR', 
        'Family_History', 'Medical_Risk', 'Cataract', 'Angle_Closure',
        'VF_Sensitivity', 'VF_Specificity', 'RNFL_Thickness', 'GCC_Thickness',
        'Corneal_Thickness', 'Symptom_Count', 'Glaucoma_Type', 'Diagnosis'
    ]
    
    processed_data = data[relevant_columns].copy()
    
    # Drop rows with missing values in critical columns
    critical_cols = ['IOP', 'CDR', 'RNFL_Thickness', 'Diagnosis']
    processed_data = processed_data.dropna(subset=critical_cols)
    
    print(f"   ✅ Processed data shape: {processed_data.shape}")
    print(f"   📊 Final diagnosis distribution: {processed_data['Diagnosis'].value_counts().to_dict()}")
    
    return processed_data

def preprocess_sensor_data(sensor_df):
    """Advanced sensor data preprocessing"""
    print("   🔧 Preprocessing sensor data...")
    
    # Ensure sensor columns exist
    sensor_cols = ['Piezo', 'FSR', 'IOP']
    for i, col in enumerate(sensor_cols):
        if col not in sensor_df.columns and i < len(sensor_df.columns):
            sensor_df[col] = sensor_df.iloc[:, i]
    
    # Create missing columns if needed with more realistic relationships
    if 'Piezo' not in sensor_df.columns:
        sensor_df['Piezo'] = np.random.uniform(0.2, 0.8, len(sensor_df))
    if 'FSR' not in sensor_df.columns:
        sensor_df['FSR'] = np.random.uniform(0.1, 0.7, len(sensor_df))
    if 'IOP' not in sensor_df.columns:
        # More complex IOP relationship
        sensor_df['IOP'] = (8 + 
                          sensor_df['Piezo'] * 18 + 
                          sensor_df['FSR'] * 12 + 
                          (sensor_df['Piezo'] * sensor_df['FSR']) * 8 +
                          np.random.normal(0, 2.5, len(sensor_df)))
        sensor_df['IOP'] = np.clip(sensor_df['IOP'], 6, 40)
    
    return sensor_df

# ==========================================
# ENHANCED FEATURE ENGINEERING
# ==========================================
def create_advanced_clinical_features(df):
    """Create advanced features using domain knowledge and statistical methods"""
    print("   🔬 Creating advanced clinical features...")
    
    # Ensure basic columns exist
    if 'Family_History' not in df.columns:
        df['Family_History'] = 0
    
    # Advanced age categorization based on glaucoma risk literature
    df['Age_Group'] = pd.cut(df['Age'], 
                            bins=[0, 40, 50, 60, 70, 80, 100], 
                            labels=[0, 1, 2, 3, 4, 5])
    df['Age_Group'] = df['Age_Group'].astype(int)
    
    # IOP categorization with more granular risk levels
    df['IOP_Category'] = pd.cut(df['IOP'], 
                               bins=[0, 10, 15, 18, 21, 25, 30, 50], 
                               labels=[0, 1, 2, 3, 4, 5, 6])
    df['IOP_Category'] = df['IOP_Category'].astype(int)
    
    # Clinical risk thresholds
    df['High_IOP'] = (df['IOP'] > 21).astype(int)
    df['Very_High_IOP'] = (df['IOP'] > 25).astype(int)
    df['Low_IOP'] = (df['IOP'] < 12).astype(int)  # Hypotony
    df['Elderly'] = (df['Age'] > 65).astype(int)
    df['High_Risk_Age'] = (df['Age'] > 50).astype(int)
    
    # CDR risk thresholds
    df['High_CDR'] = (df['CDR'] > 0.7).astype(int)
    df['Very_High_CDR'] = (df['CDR'] > 0.9).astype(int)
    df['Asymmetric_CDR'] = (df['CDR'] > 0.2).astype(int)  # Asymmetry is a risk factor
    
    # RNFL thickness risk
    df['Low_RNFL'] = (df['RNFL_Thickness'] < 80).astype(int)
    df['Very_Low_RNFL'] = (df['RNFL_Thickness'] < 70).astype(int)
    
    # GCC thickness risk
    df['Low_GCC'] = (df['GCC_Thickness'] < 70).astype(int)
    df['Very_Low_GCC'] = (df['GCC_Thickness'] < 60).astype(int)
    
    # Visual field defects
    df['VF_Defect'] = ((df['VF_Sensitivity'] < 0.7) | (df['VF_Specificity'] < 0.7)).astype(int)
    df['Severe_VF_Defect'] = ((df['VF_Sensitivity'] < 0.5) | (df['VF_Specificity'] < 0.5)).astype(int)
    
    # Corneal thickness risk
    df['Thin_Cornea'] = (df['Corneal_Thickness'] < 555).astype(int)
    df['Very_Thin_Cornea'] = (df['Corneal_Thickness'] < 500).astype(int)
    
    # Advanced interaction features
    df['Age_IOP_Interaction'] = df['Age'] * df['IOP'] / 1000
    df['Age_CDR_Interaction'] = df['Age'] * df['CDR'] / 100
    df['IOP_CDR_Interaction'] = df['IOP'] * df['CDR']
    df['Age_Family_Risk'] = df['Age'] * df['Family_History'] / 100
    df['IOP_Family_Risk'] = df['IOP'] * df['Family_History']
    df['IOP_RNFL_Interaction'] = df['IOP'] * (100 - df['RNFL_Thickness']) / 100
    df['CDR_RNFL_Interaction'] = df['CDR'] * (100 - df['RNFL_Thickness']) / 100
    
    # Comprehensive risk scoring
    risk_factors = [
        'High_IOP', 'Very_High_IOP', 'High_CDR', 'Very_High_CDR', 
        'Low_RNFL', 'Very_Low_RNFL', 'Low_GCC', 'Very_Low_GCC',
        'VF_Defect', 'Severe_VF_Defect', 'Thin_Cornea', 'Very_Thin_Cornea',
        'Angle_Closure', 'Family_History', 'High_Risk_Age'
    ]
    
    available_risk_factors = [col for col in risk_factors if col in df.columns]
    df['Medical_Risk_Score'] = df[available_risk_factors].sum(axis=1)
    
    # Combined risk categories
    df['Low_Risk'] = ((df['Age'] < 50) & (df['IOP'] < 18) & (df['CDR'] < 0.5) & 
                      (df['RNFL_Thickness'] > 85) & (df['Medical_Risk_Score'] < 2)).astype(int)
    
    df['Medium_Risk'] = ((df['Age'].between(50, 65)) | 
                         (df['IOP'].between(18, 21)) | 
                         (df['CDR'].between(0.5, 0.7)) | 
                         (df['RNFL_Thickness'].between(70, 85)) | 
                         (df['Medical_Risk_Score'].between(2, 4))).astype(int)
    
    df['High_Risk'] = ((df['Age'] > 65) | (df['IOP'] > 21) | (df['CDR'] > 0.7) | 
                       (df['RNFL_Thickness'] < 70) | (df['Medical_Risk_Score'] >= 4)).astype(int)
    
    # Statistical features
    df['IOP_Z_Score'] = (df['IOP'] - df['IOP'].mean()) / df['IOP'].std()
    df['Age_Z_Score'] = (df['Age'] - df['Age'].mean()) / df['Age'].std()
    df['CDR_Z_Score'] = (df['CDR'] - df['CDR'].mean()) / df['CDR'].std()
    df['RNFL_Z_Score'] = (df['RNFL_Thickness'] - df['RNFL_Thickness'].mean()) / df['RNFL_Thickness'].std()
    
    # Polynomial features for non-linear relationships
    df['Age_Squared'] = df['Age'] ** 2
    df['IOP_Squared'] = df['IOP'] ** 2
    df['CDR_Squared'] = df['CDR'] ** 2
    df['Age_Cubed'] = df['Age'] ** 3
    
    print(f"      ✅ Advanced features created: {df.shape[1]} total columns")
    
    return df

# ==========================================
# IMPROVED FEATURE SELECTION
# ==========================================
def select_best_features(clinical_data, max_features=20):
    """Improved feature selection with better handling of feature names"""
    print("\n⚙️ Intelligent feature selection...")
    
    df = clinical_data.copy()
    
    # Get all potential features (exclude diagnosis)
    feature_columns = [col for col in df.columns if col != 'Diagnosis']
    X = df[feature_columns]
    y = df['Diagnosis']
    
    # Fix feature names if they have np.str_ prefix
    X.columns = [str(col) for col in X.columns]
    feature_columns = [str(col) for col in feature_columns]
    
    print(f"   📊 Evaluating {len(feature_columns)} potential features")
    print(f"   📋 Class distribution: {np.bincount(y)}")
    
    # Method 1: Mutual Information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_features = np.array(feature_columns)[np.argsort(mi_scores)[-max_features:]]
    
    # Method 2: F-score
    f_selector = SelectKBest(score_func=f_classif, k=max_features)
    f_selector.fit(X, y)
    f_features = np.array(feature_columns)[f_selector.get_support()]
    
    # Method 3: Random Forest Feature Importance
    rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_temp.fit(X, y)
    rf_importance = rf_temp.feature_importances_
    rf_features = np.array(feature_columns)[np.argsort(rf_importance)[-max_features:]]
    
    # Combine selections (features that appear in multiple methods get priority)
    feature_counts = {}
    for feat in np.concatenate([mi_features, f_features, rf_features]):
        feature_counts[feat] = feature_counts.get(feat, 0) + 1
    
    # Select features that appear in at least 2 methods, then top by individual scores
    priority_features = [feat for feat, count in feature_counts.items() if count >= 2]
    
    if len(priority_features) < max_features:
        # Add remaining features by RF importance
        remaining_features = [feat for feat in feature_columns if feat not in priority_features]
        remaining_importance = {feat: rf_temp.feature_importances_[feature_columns.index(feat)] 
                              for feat in remaining_features}
        additional_features = sorted(remaining_importance.items(), key=lambda x: x[1], reverse=True)
        
        for feat, _ in additional_features:
            if len(priority_features) >= max_features:
                break
            priority_features.append(feat)
    
    selected_features = priority_features[:max_features]
    
    print(f"   📊 Selected {len(selected_features)} features")
    print(f"   📋 Features: {selected_features}")
    
    # Create a new dataframe with selected features
    selected_df = df[selected_features + ['Diagnosis']].copy()
    
    return selected_df, selected_features

# ==========================================
# IMPROVED MODEL CREATION
# ==========================================
def create_base_models():
    """Create diverse base models optimized for medical prediction with improved hyperparameters"""
    print("   🤖 Creating diverse base models...")
    
    models = {}
    
    # 1. Tree-based models with improved parameters
    models['rf'] = RandomForestClassifier(
        n_estimators=500,  # Increased
        max_depth=15,      # Increased
        min_samples_split=2,  # Decreased
        min_samples_leaf=1,   # Decreased
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    models['extra_trees'] = ExtraTreesClassifier(
        n_estimators=500,  # Increased
        max_depth=15,      # Increased
        min_samples_split=2,  # Decreased
        min_samples_leaf=1,   # Decreased
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # 2. Gradient Boosting with improved parameters
    models['gbm'] = GradientBoostingClassifier(
        n_estimators=300,  # Increased
        max_depth=10,      # Increased
        learning_rate=0.03, # Decreased
        min_samples_split=2,  # Decreased
        min_samples_leaf=1,   # Decreased
        random_state=42
    )
    
    # 3. Advanced boosting models (if available)
    if XGBOOST_AVAILABLE:
        models['xgb'] = xgb.XGBClassifier(
            n_estimators=300,  # Increased
            max_depth=10,      # Increased
            learning_rate=0.03, # Decreased
            min_child_weight=1,
            gamma=0.01,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=1,
            random_state=42,
            n_jobs=-1
        )
    
    if LIGHTGBM_AVAILABLE:
        models['lgb'] = lgb.LGBMClassifier(
            n_estimators=300,  # Increased
            max_depth=10,      # Increased
            learning_rate=0.03, # Decreased
            min_child_samples=2,  # Decreased
            min_split_gain=0.01,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight='balanced',
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
    
    if CATBOOST_AVAILABLE:
        models['catboost'] = cb.CatBoostClassifier(
            iterations=300,    # Increased
            depth=10,          # Increased
            learning_rate=0.03, # Decreased
            l2_leaf_reg=1,
            bootstrap_type='Bayesian',
            bagging_temperature=1,
            od_type='Iter',
            od_wait=50,
            random_seed=42,
            verbose=False
        )
    
    # 4. Linear models
    models['logistic'] = LogisticRegression(
        C=100.0,  # Increased
        penalty='l2',
        max_iter=3000,  # Increased
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # 5. Support Vector Machines
    models['svm_rbf'] = SVC(
        C=100.0,  # Increased
        kernel='rbf',
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=42
    )
    
    models['svm_poly'] = SVC(
        C=100.0,  # Increased
        kernel='poly',
        degree=3,
        class_weight='balanced',
        probability=True,
        random_state=42
    )
    
    # 6. Neural Network
    models['mlp'] = MLPClassifier(
        hidden_layer_sizes=(300, 150, 50),  # Increased and added layer
        activation='relu',
        solver='adam',
        alpha=0.0001,  # Decreased
        learning_rate='adaptive',
        max_iter=2000,  # Increased
        random_state=42
    )
    
    # 7. Naive Bayes (good for medical diagnosis)
    models['naive_bayes'] = GaussianNB()
    
    # 8. Discriminant Analysis
    models['lda'] = LinearDiscriminantAnalysis()
    models['qda'] = QuadraticDiscriminantAnalysis()
    
    # 9. K-Nearest Neighbors
    models['knn'] = KNeighborsClassifier(
        n_neighbors=3,  # Decreased
        weights='distance',
        metric='manhattan',
        n_jobs=-1
    )
    
    # 10. Balanced ensemble models
    models['balanced_rf'] = BalancedRandomForestClassifier(
        n_estimators=500,  # Increased
        max_depth=15,      # Increased
        min_samples_split=2,  # Decreased
        random_state=42,
        n_jobs=-1
    )
    
    models['balanced_bagging'] = BalancedBaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=15, class_weight='balanced'),  # Increased
        n_estimators=300,  # Increased
        random_state=42,
        n_jobs=-1
    )
    
    print(f"   ✅ Created {len(models)} base models")
    return models

def create_stacking_ensemble(base_models, X_train, y_train):
    """Create advanced stacking ensemble with improved model selection"""
    print("   🏗️ Building stacking ensemble...")
    
    # Select best performing base models for stacking
    best_models = []
    model_names = []
    
    # Evaluate each model with cross-validation
    model_scores = {}
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    
    # Only use models with predict_proba for stacking
    proba_models = {name: model for name, model in base_models.items() 
                   if hasattr(model, 'predict_proba')}
    
    print(f"   📊 Using {len(proba_models)} models with predict_proba for stacking")
    
    for name, model in proba_models.items():
        try:
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='balanced_accuracy', n_jobs=-1)
            model_scores[name] = scores.mean()
        except Exception as e:
            print(f"     ⚠️ Model {name} failed: {str(e)[:50]}")
            continue
    
    # Select top performing models
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    top_models = sorted_models[:8]  # Use top 8 models for stacking
    
    print(f"   📊 Model performance ranking:")
    for name, score in sorted_models:
        print(f"      {name}: {score:.4f}")
    
    # Create stacking ensemble
    for name, score in top_models:
        best_models.append((name, proba_models[name]))
        model_names.append(name)
    
    # Meta-learner
    meta_learner = LogisticRegression(
        C=10.0,
        class_weight='balanced',
        random_state=42
    )
    
    stacking_classifier = StackingClassifier(
        estimators=best_models,
        final_estimator=meta_learner,
        cv=cv,
        stack_method='predict_proba',
        n_jobs=-1
    )
    
    print(f"   ✅ Stacking ensemble created with {len(best_models)} base models")
    return stacking_classifier, model_names

def train_advanced_ensemble_classifier(clinical_data, feature_cols):
    """Train advanced multi-model ensemble with improved training process"""
    print("\n🧠 STAGE 2: Advanced Multi-Model Ensemble Classification")
    print("-" * 60)
    
    X = clinical_data[feature_cols]
    y = clinical_data['Diagnosis']
    
    print(f"   📊 Training data: {len(X)} samples, {len(feature_cols)} features")
    print(f"   📋 Class distribution: {np.bincount(y)}")
    
    # Advanced preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"   📊 Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Advanced oversampling
    if len(X_train) > 100:
        print("   ⚖️ Applying advanced SMOTE...")
        # Use BorderlineSMOTE for better boundary handling
        smote = BorderlineSMOTE(random_state=42, k_neighbors=5)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        print(f"   📊 After SMOTE: {len(X_train_balanced)} samples")
    else:
        X_train_balanced, y_train_balanced = X_train, y_train
    
    # Create base models
    base_models = create_base_models()
    
    # Create stacking ensemble
    stacking_ensemble, model_names = create_stacking_ensemble(
        base_models, X_train_balanced, y_train_balanced
    )
    
    # Train stacking ensemble
    print("   🎯 Training stacking ensemble...")
    stacking_ensemble.fit(X_train_balanced, y_train_balanced)
    
    # Create voting ensemble as backup
    # Only use models with predict_proba for soft voting
    voting_models = [(name, base_models[name]) for name in model_names[:5] 
                     if hasattr(base_models[name], 'predict_proba')]
    
    if voting_models:
        voting_ensemble = VotingClassifier(
            estimators=voting_models,
            voting='soft'
        )
        voting_ensemble.fit(X_train_balanced, y_train_balanced)
    else:
        voting_ensemble = None
    
    # Evaluate both ensembles
    print("\n   📈 Model Performance Evaluation:")
    print("   " + "-" * 50)
    
    # Cross-validation for stacking ensemble
    cv_scores_stacking = cross_val_score(
        stacking_ensemble, X_train_balanced, y_train_balanced,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring='balanced_accuracy',
        n_jobs=-1
    )
    
    if voting_ensemble:
        # Cross-validation for voting ensemble
        cv_scores_voting = cross_val_score(
            voting_ensemble, X_train_balanced, y_train_balanced,
            cv=StratifiedKFold(5, shuffle=True, random_state=42),
            scoring='balanced_accuracy',
            n_jobs=-1
        )
        
        print(f"   🔄 Stacking Ensemble CV: {cv_scores_stacking.mean():.3f} ± {cv_scores_stacking.std():.3f}")
        print(f"   🔄 Voting Ensemble CV: {cv_scores_voting.mean():.3f} ± {cv_scores_voting.std():.3f}")
        
        # Choose best ensemble
        best_ensemble = stacking_ensemble if cv_scores_stacking.mean() > cv_scores_voting.mean() else voting_ensemble
        ensemble_type = "Stacking" if cv_scores_stacking.mean() > cv_scores_voting.mean() else "Voting"
    else:
        print(f"   🔄 Stacking Ensemble CV: {cv_scores_stacking.mean():.3f} ± {cv_scores_stacking.std():.3f}")
        best_ensemble = stacking_ensemble
        ensemble_type = "Stacking"
    
    print(f"   🏆 Selected: {ensemble_type} Ensemble")
    
    # Test performance
    test_pred = best_ensemble.predict(X_test)
    test_proba = best_ensemble.predict_proba(X_test)[:, 1]
    
    # Comprehensive metrics
    test_accuracy = accuracy_score(y_test, test_pred)
    test_balanced_acc = balanced_accuracy_score(y_test, test_pred)
    test_precision = precision_score(y_test, test_pred, zero_division=0)
    test_recall = recall_score(y_test, test_pred, zero_division=0)
    test_f1 = f1_score(y_test, test_pred, zero_division=0)
    test_auc = roc_auc_score(y_test, test_proba)
    test_avg_precision = average_precision_score(y_test, test_proba)
    
    print(f"\n   🎯 Final {ensemble_type} Ensemble Performance:")
    print(f"      Accuracy: {test_accuracy:.3f}")
    print(f"      Balanced Accuracy: {test_balanced_acc:.3f}")
    print(f"      Precision: {test_precision:.3f}")
    print(f"      Recall: {test_recall:.3f}")
    print(f"      F1-Score: {test_f1:.3f}")
    print(f"      AUC-ROC: {test_auc:.3f}")
    print(f"      AUC-PR: {test_avg_precision:.3f}")
    
    # Detailed classification report
    print(f"\n   📋 Detailed Classification Report:")
    print(classification_report(y_test, test_pred, 
                              target_names=['Normal', 'Glaucoma'],
                              digits=3))
    
    # Feature importance analysis
    try:
        if hasattr(best_ensemble, 'feature_importances_'):
            feature_importance = best_ensemble.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)
            
            print(f"\n   📊 Top 10 Feature Importances:")
            for idx, row in importance_df.head(10).iterrows():
                print(f"      {row['Feature']}: {row['Importance']:.4f}")
    except:
        pass
    
    return best_ensemble, scaler, ensemble_type

# ==========================================
# PREDICTION FUNCTION
# ==========================================
def predict_glaucoma(patient_data, sensor_data=None):
    """
    Predict glaucoma using trained models
    
    Parameters:
    - patient_data: DataFrame with clinical features
    - sensor_data: DataFrame with sensor readings (optional)
    
    Returns:
    - Dictionary with prediction results
    """
    try:
        # Load models
        with open('advanced_glaucoma_classifier.pkl', 'rb') as f:
            classifier = pickle.load(f)
        with open('advanced_clinical_scaler.pkl', 'rb') as f:
            clinical_scaler = pickle.load(f)
        with open('feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        with open('ensemble_type.pkl', 'rb') as f:
            ensemble_type = pickle.load(f)
        
        # Process IOP if sensor data provided
        if sensor_data is not None:
            with open('advanced_iop_regressor.pkl', 'rb') as f:
                iop_regressor = pickle.load(f)
            with open('advanced_iop_scaler.pkl', 'rb') as f:
                iop_scaler = pickle.load(f)
            
            # Prepare sensor data
            X_sensor = sensor_data[['Piezo', 'FSR']].values
            X_sensor_scaled = iop_scaler.transform(X_sensor)
            
            # Predict IOP
            predicted_iop = iop_regressor.predict(X_sensor_scaled)[0]
            patient_data['IOP'] = predicted_iop
        
        # Ensure all required features exist
        for feature in feature_columns:
            if feature not in patient_data.columns:
                # Create missing features with default values
                if feature == 'Age_Group':
                    patient_data['Age_Group'] = pd.cut(patient_data['Age'], 
                                                      bins=[0, 40, 50, 60, 70, 80, 100], 
                                                      labels=[0, 1, 2, 3, 4, 5]).astype(int)
                elif feature == 'IOP_Category':
                    patient_data['IOP_Category'] = pd.cut(patient_data['IOP'], 
                                                         bins=[0, 10, 15, 18, 21, 25, 30, 50], 
                                                         labels=[0, 1, 2, 3, 4, 5, 6]).astype(int)
                elif feature == 'High_IOP':
                    patient_data['High_IOP'] = (patient_data['IOP'] > 21).astype(int)
                elif feature == 'Very_High_IOP':
                    patient_data['Very_High_IOP'] = (patient_data['IOP'] > 25).astype(int)
                elif feature == 'Low_IOP':
                    patient_data['Low_IOP'] = (patient_data['IOP'] < 12).astype(int)
                elif feature == 'Elderly':
                    patient_data['Elderly'] = (patient_data['Age'] > 65).astype(int)
                elif feature == 'High_Risk_Age':
                    patient_data['High_Risk_Age'] = (patient_data['Age'] > 50).astype(int)
                elif feature == 'High_CDR':
                    patient_data['High_CDR'] = (patient_data['CDR'] > 0.7).astype(int)
                elif feature == 'Very_High_CDR':
                    patient_data['Very_High_CDR'] = (patient_data['CDR'] > 0.9).astype(int)
                elif feature == 'Low_RNFL':
                    patient_data['Low_RNFL'] = (patient_data['RNFL_Thickness'] < 80).astype(int)
                elif feature == 'Very_Low_RNFL':
                    patient_data['Very_Low_RNFL'] = (patient_data['RNFL_Thickness'] < 70).astype(int)
                elif feature == 'VF_Defect':
                    patient_data['VF_Defect'] = ((patient_data['VF_Sensitivity'] < 0.7) | 
                                                 (patient_data['VF_Specificity'] < 0.7)).astype(int)
                elif feature == 'Thin_Cornea':
                    patient_data['Thin_Cornea'] = (patient_data['Corneal_Thickness'] < 555).astype(int)
                elif feature == 'Medical_Risk_Score':
                    risk_factors = [
                        'High_IOP', 'Very_High_IOP', 'High_CDR', 'Very_High_CDR', 
                        'Low_RNFL', 'Very_Low_RNFL', 'VF_Defect', 'Thin_Cornea',
                        'Angle_Closure', 'Family_History', 'High_Risk_Age'
                    ]
                    available_risk_factors = [col for col in risk_factors if col in patient_data.columns]
                    patient_data['Medical_Risk_Score'] = patient_data[available_risk_factors].sum(axis=1)
                elif feature == 'Low_Risk':
                    patient_data['Low_Risk'] = ((patient_data['Age'] < 50) & 
                                               (patient_data['IOP'] < 18) & 
                                               (patient_data['CDR'] < 0.5) & 
                                               (patient_data['RNFL_Thickness'] > 85) & 
                                               (patient_data['Medical_Risk_Score'] < 2)).astype(int)
                elif feature == 'Medium_Risk':
                    patient_data['Medium_Risk'] = ((patient_data['Age'].between(50, 65)) | 
                                                  (patient_data['IOP'].between(18, 21)) | 
                                                  (patient_data['CDR'].between(0.5, 0.7)) | 
                                                  (patient_data['RNFL_Thickness'].between(70, 85)) | 
                                                  (patient_data['Medical_Risk_Score'].between(2, 4))).astype(int)
                elif feature == 'High_Risk':
                    patient_data['High_Risk'] = ((patient_data['Age'] > 65) | 
                                                (patient_data['IOP'] > 21) | 
                                                (patient_data['CDR'] > 0.7) | 
                                                (patient_data['RNFL_Thickness'] < 70) | 
                                                (patient_data['Medical_Risk_Score'] >= 4)).astype(int)
                elif feature == 'IOP_Z_Score':
                    patient_data['IOP_Z_Score'] = (patient_data['IOP'] - 15.5) / 3.0  # Approximate mean and std
                elif feature == 'Age_Z_Score':
                    patient_data['Age_Z_Score'] = (patient_data['Age'] - 60) / 15  # Approximate mean and std
                elif feature == 'CDR_Z_Score':
                    patient_data['CDR_Z_Score'] = (patient_data['CDR'] - 0.6) / 0.2  # Approximate mean and std
                elif feature == 'RNFL_Z_Score':
                    patient_data['RNFL_Z_Score'] = (patient_data['RNFL_Thickness'] - 80) / 10  # Approximate mean and std
                elif feature == 'Age_Squared':
                    patient_data['Age_Squared'] = patient_data['Age'] ** 2
                elif feature == 'IOP_Squared':
                    patient_data['IOP_Squared'] = patient_data['IOP'] ** 2
                elif feature == 'CDR_Squared':
                    patient_data['CDR_Squared'] = patient_data['CDR'] ** 2
                elif feature == 'Age_Cubed':
                    patient_data['Age_Cubed'] = patient_data['Age'] ** 3
                elif feature == 'Age_IOP_Interaction':
                    patient_data['Age_IOP_Interaction'] = patient_data['Age'] * patient_data['IOP'] / 1000
                elif feature == 'Age_CDR_Interaction':
                    patient_data['Age_CDR_Interaction'] = patient_data['Age'] * patient_data['CDR'] / 100
                elif feature == 'IOP_CDR_Interaction':
                    patient_data['IOP_CDR_Interaction'] = patient_data['IOP'] * patient_data['CDR']
                elif feature == 'Age_Family_Risk':
                    patient_data['Age_Family_Risk'] = patient_data['Age'] * patient_data['Family_History'] / 100
                elif feature == 'IOP_Family_Risk':
                    patient_data['IOP_Family_Risk'] = patient_data['IOP'] * patient_data['Family_History']
                elif feature == 'IOP_RNFL_Interaction':
                    patient_data['IOP_RNFL_Interaction'] = patient_data['IOP'] * (100 - patient_data['RNFL_Thickness']) / 100
                elif feature == 'CDR_RNFL_Interaction':
                    patient_data['CDR_RNFL_Interaction'] = patient_data['CDR'] * (100 - patient_data['RNFL_Thickness']) / 100
                else:
                    patient_data[feature] = 0  # Default value for unknown features
        
        # Select features and scale
        X = patient_data[feature_columns]
        X_scaled = clinical_scaler.transform(X)
        
        # Make prediction
        prediction = classifier.predict(X_scaled)[0]
        probability = classifier.predict_proba(X_scaled)[0, 1]
        
        # Prepare result
        result = {
            'prediction': 'Glaucoma' if prediction == 1 else 'Normal',
            'probability': float(probability),
            'confidence': 'High' if probability > 0.8 or probability < 0.2 else 'Medium',
            'ensemble_type': ensemble_type,
            'features_used': feature_columns,
            'iop_value': patient_data['IOP'].values[0] if 'IOP' in patient_data.columns else None
        }
        
        return result
    
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return None

# ==========================================
# MAIN ADVANCED PIPELINE
# ==========================================
def main():
    """Advanced main pipeline with multi-model ensemble"""
    print("🎯 Starting ADVANCED Training Pipeline...")
    
    try:
        # Load and preprocess data
        clinical_data, sensor_data = load_and_preprocess_data()
        
        if clinical_data is None or sensor_data is None:
            print("❌ Cannot proceed without data")
            return False
        
        print(f"✅ Data loaded: {len(clinical_data)} clinical, {len(sensor_data)} sensor samples")
        
        # Create advanced features
        print("\n" + "="*70)
        clinical_data = create_advanced_clinical_features(clinical_data)
        
        # Intelligent feature selection
        print("\n" + "="*70)
        clinical_processed, feature_cols = select_best_features(clinical_data, max_features=20)
        
        # Train regularized IOP regressor (keep existing)
        print("\n" + "="*70)
        from sklearn.linear_model import Ridge
        
        # Simple IOP regressor for sensor data
        X_sensor = sensor_data[['Piezo', 'FSR']].values
        y_sensor = sensor_data['IOP'].values
        
        iop_scaler = StandardScaler()
        X_sensor_scaled = iop_scaler.fit_transform(X_sensor)
        
        iop_regressor = Ridge(alpha=1.0, random_state=42)
        iop_regressor.fit(X_sensor_scaled, y_sensor)
        
        print("🤖 STAGE 1: Simple IOP Regression Model")
        print(f"   ✅ Ridge regression trained on {len(X_sensor)} samples")
        
        # Train advanced ensemble classifier  
        print("\n" + "="*70)
        glaucoma_classifier, clinical_scaler, ensemble_type = train_advanced_ensemble_classifier(
            clinical_processed, feature_cols
        )
        
        # Save all models
        print("\n💾 Saving advanced models...")
        models_to_save = {
            'advanced_iop_regressor': iop_regressor,
            'advanced_iop_scaler': iop_scaler,
            'advanced_glaucoma_classifier': glaucoma_classifier,
            'advanced_clinical_scaler': clinical_scaler,
            'feature_columns': feature_cols,
            'ensemble_type': ensemble_type
        }
        
        for name, model in models_to_save.items():
            try:
                filename = f'{name}.pkl'
                with open(filename, 'wb') as file:
                    pickle.dump(model, file)
                print(f"   ✅ Saved: {filename}")
            except Exception as e:
                print(f"   ❌ Error saving {name}: {e}")
        
        print("\n🎉 Advanced training pipeline completed successfully!")
        print(f"   📁 Saved {len(models_to_save)} model files")
        
        # Demonstrate prediction
        print("\n🔮 Demonstrating prediction with sample data...")
        sample_patient = clinical_processed[feature_cols].iloc[0:1].copy()
        sample_sensor = sensor_data.iloc[0:1].copy()
        
        # Make prediction
        prediction_result = predict_glaucoma(sample_patient, sample_sensor)
        
        if prediction_result:
            print(f"\n👤 Sample Patient Prediction:")
            print(f"   Prediction: {prediction_result['prediction']}")
            print(f"   Probability: {prediction_result['probability']:.3f}")
            print(f"   Confidence: {prediction_result['confidence']}")
            print(f"   IOP Value: {prediction_result['iop_value']:.2f} mmHg")
            print(f"   Model Used: {prediction_result['ensemble_type']} Ensemble")
        else:
            print("   ❌ Prediction failed")
        
        return True
    
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        return False

if __name__ == '__main__':
    success = main()
    if success:
        print("\n✅ All operations completed successfully!")
    else:
        print("\n❌ Some operations failed!")