import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.metrics import classification_report, precision_recall_curve, roc_curve, auc
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  
    return df

def preprocess_dataset(df, is_training=True, verbose=True):
    """
    Preprocess the dataset with feature engineering.
    
    Parameters:
    df (DataFrame): Dataset to preprocess
    is_training (bool): Whether this is training data (with Diagnosis) or prediction data
    verbose (bool): Whether to print information
    
    Returns:
    DataFrame: Preprocessed dataframe
    """
    # Check required columns based on whether we're preprocessing training or prediction data
    if is_training:
        expected_columns = ['Age', 'Gender', 'IOP', 'Diagnosis']
    else:
        expected_columns = ['Age', 'Gender', 'IOP']
    
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        if verbose:
            print(f"Warning: Missing columns in dataset: {missing_cols}")
            print("Will attempt to use default values for missing columns")
        
        # Add default values for missing columns
        for col in missing_cols:
            if col == 'Age':
                df['Age'] = 20  # Default age
            elif col == 'Gender':
                df['Gender'] = 0  # Default gender (0 = female, 1 = male)
            elif col == 'IOP' and 'Piezo' in df.columns and 'FSR' in df.columns:
                # If we have sensor data but no IOP, we might need to calculate it
                # This is placeholder logic - replace with actual conversion formula if available
                df['IOP'] = df['Piezo'] * 6.06  # Example conversion
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # For training data, drop duplicates and missing values
    if is_training:
        initial_rows = len(df)
        # Focus only on required columns for training
        df = df[expected_columns].drop_duplicates().dropna()
        if verbose:
            print(f"Removed {initial_rows - len(df)} duplicate or missing rows")
    
    # Create feature engineering with reduced complexity
    # Basic transformations
    df['Age_IOP'] = df['Age'] * df['IOP']
    df['IOP_squared'] = df['IOP'] ** 2
    df['IOP_sqrt'] = np.sqrt(np.maximum(1, df['IOP']))
    df['IOP_log'] = np.log1p(df['IOP'])
    
    # Ratios and differences
    df['Age_IOP_ratio'] = df['Age'] / (df['IOP'] + 1)
    df['IOP_Age_ratio'] = df['IOP'] / (df['Age'] + 1)
    df['Age_IOP_diff'] = df['Age'] - df['IOP']
    
    # Use fixed bins to avoid bin edge issues
    age_bins = [0, 20, 40, 60, 80, float('inf')]
    iop_bins = [0, 10, 15, 20, 25, float('inf')]
    
    try:
        # Only bin if there's sufficient variance in the data
        if df['Age'].nunique() > 1:
            df['Age_bin'] = pd.cut(df['Age'], bins=age_bins, labels=False, duplicates='drop')
        else:
            df['Age_bin'] = 2  # Middle bin as default
            
        if df['IOP'].nunique() > 1:
            df['IOP_bin'] = pd.cut(df['IOP'], bins=iop_bins, labels=False, duplicates='drop')
        else:
            df['IOP_bin'] = 2  # Middle bin as default
    except Exception as e:
        if verbose:
            print(f"Warning during binning: {e}")
        # Fallback to simpler approach
        df['Age_bin'] = 2  # Middle bin as default
        df['IOP_bin'] = 2  # Middle bin as default
    
    # Interaction terms - always create this feature
    df['Age_bin_IOP_bin'] = df['Age_bin'] * df['IOP_bin']
    
    # Advanced statistical features
    if df['IOP'].nunique() > 1:
        df['z_IOP'] = (df['IOP'] - df['IOP'].mean()) / (df['IOP'].std() + 1e-8)
    else:
        df['z_IOP'] = 0  # Default when no variance
    
    # Simple trigonometric transformations
    df['IOP_sin'] = np.sin(df['IOP'] / 5)
    df['IOP_cos'] = np.cos(df['IOP'] / 5)
    
    # Encode categorical variables
    if df['Gender'].dtype == object:
        df['Gender'] = LabelEncoder().fit_transform(df['Gender'])  
    
    # For training data, ensure diagnosis is binary
    if is_training and 'Diagnosis' in df.columns:
        df['Diagnosis'] = df['Diagnosis'].astype(int)
    
    # Add outlier detection - replace extreme values with winsorization
    for col in ['Age', 'IOP', 'Age_IOP', 'IOP_squared']:
        if col in df.columns:
            q1 = df[col].quantile(0.01)
            q3 = df[col].quantile(0.99)
            df[col] = df[col].clip(q1, q3)
    
    if verbose:
        print(f"Final dataset shape: {df.shape}")
        if is_training:
            print(f"Features created: {len(df.columns) - len(expected_columns)}")
    
    return df

# Function to process sensor data
def process_sensor_data(sensor_data, demographic_info=None):
    """
    Process raw sensor data and prepare it for prediction
    
    Parameters:
    sensor_data (str or DataFrame): Either a path to CSV file or DataFrame with sensor readings
    demographic_info (dict): Dictionary with 'Age' and 'Gender' keys
    
    Returns:
    DataFrame: Processed data ready for prediction
    """
    # Set default demographic info if not provided
    if demographic_info is None:
        demographic_info = {'Age': 20, 'Gender': 0}
    
    # Load data if it's a path
    if isinstance(sensor_data, str):
        try:
            data = pd.read_csv(sensor_data)
        except:
            # Try parsing as direct data
            try:
                lines = sensor_data.strip().split('\n')
                rows = [line.split(',') for line in lines]
                data = pd.DataFrame(rows, columns=['Piezo', 'FSR', 'IOP'])
                data = data.astype(float)
            except:
                raise ValueError("Could not parse sensor data as CSV or direct data")
    else:
        data = sensor_data.copy()
    
    # Ensure column names are standardized
    data.columns = data.columns.str.strip()
    
    # Check if we need to calculate IOP from sensor data
    if 'IOP' not in data.columns and 'Piezo' in data.columns and 'FSR' in data.columns:
        # This is a placeholder formula - replace with actual conversion if available
        data['IOP'] = data['Piezo'] * 6.06  # Example conversion
    
    # Add demographic information to each row
    for col, value in demographic_info.items():
        data[col] = value
    
    # Preprocess the data
    processed_data = preprocess_dataset(data, is_training=False, verbose=False)
    
    return processed_data

# Main function to train model
def train_model(dataset_path):
    print("Loading and preprocessing data...")
    df = load_dataset(dataset_path)
    df = preprocess_dataset(df, is_training=True)

    # Define features and target
    all_features = [col for col in df.columns if col != 'Diagnosis']
    X = df[all_features]
    y = df['Diagnosis']

    # Feature selection to reduce dimensionality and avoid overfitting
    print("Performing feature selection...")
    selector = SelectFromModel(
        ExtraTreesClassifier(n_estimators=100, random_state=42),
        max_features=10  # Reduced from 15 to prevent overfitting
    )
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()]
    X_selected = X[selected_features]

    print(f"Selected features: {selected_features.tolist()}")

    # Check class distribution before resampling
    class_counts = np.bincount(y)
    print(f"Class distribution before resampling: {class_counts}")

    # Modified resampling approach with better validation
    print("Applying data resampling techniques...")
    resamplers = {
        'SMOTE': SMOTE(random_state=42, sampling_strategy='auto'),
        'BorderlineSMOTE': BorderlineSMOTE(random_state=42, sampling_strategy='auto'),
        'SMOTETomek': SMOTETomek(random_state=42, sampling_strategy='auto'),
        'SMOTEENN': SMOTEENN(random_state=42, sampling_strategy='auto')
    }

    # Create a stratified k-fold for consistent evaluation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_resampler = None
    best_cv_score = 0

    for name, resampler in resamplers.items():
        try:
            print(f"Testing resampler: {name}")
            X_res, y_res = resampler.fit_resample(X_selected, y)
            
            # Quick evaluation
            clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            cv_score = np.mean(cross_val_score(clf, X_res, y_res, cv=skf))
            print(f"  CV Score: {cv_score:.4f}")
            
            if cv_score > best_cv_score:
                best_cv_score = cv_score
                best_resampler = resampler
                best_name = name
        except Exception as e:
            print(f"  Error with {name}: {e}")
            continue

    if best_resampler is None:
        print("All resamplers failed. Using original data.")
        X_resampled, y_resampled = X_selected, y
    else:
        print(f"Best resampler: {best_name} with CV score: {best_cv_score:.4f}")
        X_resampled, y_resampled = best_resampler.fit_resample(X_selected, y)
        # Check class distribution after resampling
        resampled_class_counts = np.bincount(y_resampled)
        print(f"Class distribution after resampling: {resampled_class_counts}")

    # Split dataset with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )

    # Feature scaling - trying different scalers
    scalers = {
        'Standard': StandardScaler(),
        'MinMax': MinMaxScaler(),
        'Robust': RobustScaler(),
        'Power': PowerTransformer()
    }

    best_scaler = None
    best_scaler_score = 0

    for name, scaler in scalers.items():
        print(f"Testing scaler: {name}")
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Quick evaluation
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        clf.fit(X_train_scaled, y_train)
        score = clf.score(X_test_scaled, y_test)
        print(f"  Test Score: {score:.4f}")
        
        if score > best_scaler_score:
            best_scaler_score = score
            best_scaler = scaler
            best_scaler_name = name

    print(f"Best scaler: {best_scaler_name} with Test score: {best_scaler_score:.4f}")
    X_train_scaled = best_scaler.fit_transform(X_train)
    X_test_scaled = best_scaler.transform(X_test)

    # PCA for additional feature creation
    n_components = min(3, len(selected_features))
    if n_components < 2:
        n_components = 2

    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Combine original scaled features with PCA features
    X_train_combined = np.hstack((X_train_scaled, X_train_pca))
    X_test_combined = np.hstack((X_test_scaled, X_test_pca))

    # Define base models with more regularization to prevent overfitting
    base_models = {
        'RF': RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced',
            random_state=42
        ),
        'SVM': SVC(
            kernel='rbf', 
            C=100,
            gamma='scale', 
            probability=True, 
            class_weight='balanced',
            random_state=42
        ),
        'GB': GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            min_samples_split=5,
            subsample=0.7,
            max_features=0.7,
            random_state=42
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            p=1,
            n_jobs=-1
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(50, 25),
            activation='relu',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=1000,
            random_state=42
        ),
        'ET': ExtraTreesClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42
        ),
        'LR': LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='liblinear',
            class_weight='balanced',
            random_state=42
        )
    }

    # Train and evaluate base models
    base_results = {}

    print("\nBase Model Performance:")
    print("-" * 50)

    for model_name, model in base_models.items():
        print(f"Training {model_name}...")
        model.fit(X_train_combined, y_train)
        
        train_accuracy = model.score(X_train_combined, y_train)
        test_accuracy = model.score(X_test_combined, y_test)
        
        base_results[model_name] = {'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy}
        print(f"{model_name}: Train Acc = {train_accuracy:.4f}, Test Acc = {test_accuracy:.4f}")

    # Create ensemble models
    print("\nBuilding ensemble models...")

    # 1. Voting Classifier
    voting_hard = VotingClassifier(
        estimators=[(name, model) for name, model in base_models.items()],
        voting='hard'
    )
    voting_soft = VotingClassifier(
        estimators=[(name, model) for name, model in base_models.items()],
        voting='soft'
    )

    # 2. Stacking Classifier
    stacking = StackingClassifier(
        estimators=[(name, model) for name, model in base_models.items()],
        final_estimator=LogisticRegression(C=0.5, class_weight='balanced'),
        cv=5
    )

    # Dictionary of ensemble models
    ensemble_models = {
        'VotingHard': voting_hard,
        'VotingSoft': voting_soft,
        'Stacking': stacking
    }

    # Train and evaluate ensemble models
    ensemble_results = {}

    print("\nEnsemble Model Performance:")
    print("-" * 50)

    for model_name, model in ensemble_models.items():
        print(f"Training {model_name}...")
        model.fit(X_train_combined, y_train)
        
        train_accuracy = model.score(X_train_combined, y_train)
        test_accuracy = model.score(X_test_combined, y_test)
        
        ensemble_results[model_name] = {'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy}
        print(f"{model_name}: Train Acc = {train_accuracy:.4f}, Test Acc = {test_accuracy:.4f}")
        
        if test_accuracy >= 0.8:
            print(f"✓ {model_name} achieved target accuracy!")
        
        # Detailed evaluation
        y_pred = model.predict(X_test_combined)
        print(classification_report(y_test, y_pred))

    # Find best model among all (base + ensemble)
    all_results = {**base_results, **ensemble_results}
    best_model_name = max(all_results, key=lambda x: all_results[x]['test_accuracy'])

    if best_model_name in base_models:
        best_model = base_models[best_model_name]
    else:
        best_model = ensemble_models[best_model_name]

    print(f"\nBest Model: {best_model_name}")
    print(f"Training Accuracy: {all_results[best_model_name]['train_accuracy']:.4f}")
    print(f"Test Accuracy: {all_results[best_model_name]['test_accuracy']:.4f}")

    # Check for overfitting
    train_test_diff = all_results[best_model_name]['train_accuracy'] - all_results[best_model_name]['test_accuracy']
    if train_test_diff > 0.1:
        print(f"Warning: Potential overfitting detected. Train-test accuracy gap: {train_test_diff:.4f}")
        print("Consider additional regularization or reducing model complexity.")

    # Save the trained components for prediction
    print("\nSaving model components...")
    model_components = {
        'feature_selector': selector,
        'selected_features': selected_features,
        'scaler': best_scaler,
        'pca': pca,
        'model': best_model,
        'all_features': all_features
    }
    
    with open('glaucoma_model_components.pkl', 'wb') as f:
        pickle.dump(model_components, f)
    
    print("Model components saved successfully.")
    
    return model_components

# Function to predict glaucoma from sensor readings
def predict_glaucoma(sensor_data, demographic_info=None, model_components=None):
    """
    Predict glaucoma from sensor readings
    
    Parameters:
    sensor_data (str or DataFrame): Either a path to CSV file, raw data string, or DataFrame with sensor readings
    demographic_info (dict): Dictionary with 'Age' and 'Gender' keys
    model_components (dict): Optional dictionary with trained model components
    
    Returns:
    tuple: (prediction, confidence, details)
    """
    # Set default demographic info if not provided
    if demographic_info is None:
        demographic_info = {'Age': 50, 'Gender': 0}
    
    # Load model components if not provided
    if model_components is None:
        try:
            with open('glaucoma_model_components.pkl', 'rb') as f:
                model_components = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Model components file not found. Please train the model first.")
    
    # Extract model components
    feature_selector = model_components['feature_selector']
    selected_features = model_components['selected_features']
    scaler = model_components['scaler']
    pca = model_components['pca']
    model = model_components['model']
    
    # Process sensor data
    try:
        # Handle string input (direct data)
        if isinstance(sensor_data, str) and not sensor_data.endswith('.csv'):
            # Try to parse as direct data
            try:
                lines = sensor_data.strip().split('\n')
                rows = [line.split(',') for line in lines]
                data_df = pd.DataFrame(rows, columns=['Piezo', 'FSR', 'IOP'])
                data_df = data_df.astype(float)
            except:
                raise ValueError("Could not parse sensor data as direct data")
        # Handle DataFrame or CSV file
        else:
            if isinstance(sensor_data, str):
                data_df = pd.read_csv(sensor_data)
            else:
                data_df = sensor_data.copy()
        
        # Process the data
        processed_data = process_sensor_data(data_df, demographic_info)
        
        # Ensure we have all the selected features
        missing_features = [f for f in selected_features if f not in processed_data.columns]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            # Add missing features with appropriate default values
            for feature in missing_features:
                if 'bin' in feature:
                    processed_data[feature] = 2  # Middle bin as default
                elif feature == 'z_IOP':
                    processed_data[feature] = 0  # Default z-score
                elif 'IOP' in feature:
                    if 'IOP' in processed_data.columns:
                        # Try to calculate the feature
                        iop_value = processed_data['IOP'].mean()
                        if feature == 'IOP_squared':
                            processed_data[feature] = iop_value ** 2
                        elif feature == 'IOP_sqrt':
                            processed_data[feature] = np.sqrt(max(1, iop_value))
                        elif feature == 'IOP_log':
                            processed_data[feature] = np.log1p(iop_value)
                        elif feature == 'IOP_sin':
                            processed_data[feature] = np.sin(iop_value / 5)
                        elif feature == 'IOP_cos':
                            processed_data[feature] = np.cos(iop_value / 5)
                        else:
                            processed_data[feature] = 0  # Default value
                    else:
                        processed_data[feature] = 0  # Default value
                else:
                    processed_data[feature] = 0  # Default value
        
        # Extract feature values
        X = processed_data[selected_features].values
        
        # Scale the features
        X_scaled = scaler.transform(X)
        
        # Apply PCA transformation
        X_pca = pca.transform(X_scaled)
        
        # Combine original and PCA features
        X_combined = np.hstack((X_scaled, X_pca))
        
        # Make prediction
        prediction_probs = model.predict_proba(X_combined)
        
        # Calculate both probability-based and voting-based results
        # Calculate both probability-based and voting-based results
        avg_prob = np.mean(prediction_probs, axis=0)
        prob_based_class = 1 if avg_prob[1] > 0.5 else 0
        prob_confidence = avg_prob[prob_based_class]

        # Create individual predictions using threshold of 0.5
        individual_predictions = [1 if prob[1] > 0.5 else 0 for prob in prediction_probs]

        # Use majority voting for final prediction
        vote_based_class = 1 if sum(individual_predictions) > len(individual_predictions)/2 else 0

        # Calculate confidence as the proportion of readings that match the final class
        if vote_based_class == 1:
            vote_confidence = sum(individual_predictions) / len(individual_predictions)
        else:
            vote_confidence = (len(individual_predictions) - sum(individual_predictions)) / len(individual_predictions)

        # Use majority voting as the primary method
        final_class = vote_based_class
        confidence = vote_confidence
        
        # Create readable labels for individual readings
        reading_analysis = []
        for i, prob in enumerate(prediction_probs):
            label = "Glaucoma" if prob[1] > 0.5 else "Normal"
            reading_analysis.append(f"Reading {i+1}: {label} (Probability: {prob[1]:.2f})")
        
        # Prepare detailed results
        details = {
            'individual_probabilities': prediction_probs,
            'average_probability': avg_prob,
            'probability_based': {'prediction': prob_based_class, 'confidence': prob_confidence},
            'voting_based': {'prediction': vote_based_class, 'confidence': vote_confidence},
            'num_readings': len(processed_data),
            'reading_analysis': reading_analysis,
            'feature_values': {
                feature: processed_data[feature].mean() 
                for feature in selected_features
                if feature in processed_data.columns
            }
        }
        
        return final_class, confidence, details
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        # Default to no glaucoma with low confidence
        return 0, 0.5, {'error': str(e)}
    
    
# Main function to run the program
def main():
    # Check if we need to train or load model
    try:
        with open('glaucoma_model_components.pkl', 'rb') as f:
            model_components = pickle.load(f)
        print("Loaded existing model components")
    except FileNotFoundError:
        print("No existing model found. Training new model...")
        try:
            model_components = train_model('glaucoma_dataset.csv')
        except FileNotFoundError:
            print("Error: Training dataset not found. Please provide glaucoma_dataset.csv")
            return
    
    # Read sensor data from CSV file
    sensor_file = 'sensor_data.csv'  # Change this to your sensor data file name
    try:
        sensor_df = pd.read_csv(sensor_file)
        print(f"Loaded sensor data from {sensor_file}")
    except FileNotFoundError:
        print(f"Error: Sensor data file {sensor_file} not found.")
        return
    
    # Set demographic information
    demographic_info = {'Age': 55, 'Gender': 1}  # Example: 55-year-old male
    
    # Make prediction
    prediction, confidence, details = predict_glaucoma(sensor_df, demographic_info, model_components)
    
    # Print individual reading probabilities for transparency
    if 'individual_probabilities' in details:
        print("\nIndividual reading analysis:")
        for i, probs in enumerate(details['individual_probabilities']):
            reading_result = "Glaucoma" if probs[1] > 0.5 else "Normal"
            print(f"Reading {i+1}: {reading_result} (Probability: {probs[1]:.2f})")
    
    # Output results
    result = "Glaucoma" if prediction == 1 else "No Glaucoma"
    print(f"\n=== Glaucoma Prediction Results ===")
    print(f"Prediction: {result}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Number of readings analyzed: {details['num_readings']}")

    # Print feature importance if available
    # if 'feature_values' in details:
    #     print("\nKey measurements:")
    #     iop_value = details['feature_values'].get('IOP', 'N/A')
    #     if isinstance(iop_value, (int, float)):
    #         print(f"Average IOP: {iop_value:.2f} mmHg")
    #     else:
    #         print(f"Average IOP: {iop_value}")
    
if __name__ == "__main__":
    main()