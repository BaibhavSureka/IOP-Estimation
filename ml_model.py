import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import pickle

# Load dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  
    return df

def preprocess_dataset(df):
    expected_columns = ['Age', 'Gender', 'IOP', 'Diagnosis']
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in dataset: {missing_cols}")
    
    # Drop missing values
    df = df[expected_columns].dropna()
    
    # Create feature interactions and transformations
    df['Age_IOP'] = df['Age'] * df['IOP']
    df['Age_squared'] = df['Age'] ** 2
    df['IOP_squared'] = df['IOP'] ** 2
    df['log_Age'] = np.log1p(df['Age'])
    df['log_IOP'] = np.log1p(df['IOP'])
    
    # Encode categorical variables
    if df['Gender'].dtype == object:
        df['Gender'] = LabelEncoder().fit_transform(df['Gender'])  
    
    # Ensure diagnosis is binary
    df['Diagnosis'] = df['Diagnosis'].astype(int)
    
    # Add outlier detection - replace extreme values with boundaries
    for col in ['Age', 'IOP', 'Age_IOP', 'Age_squared', 'IOP_squared']:
        q1 = df[col].quantile(0.01)
        q3 = df[col].quantile(0.99)
        df[col] = df[col].clip(q1, q3)
    
    return df

# Load and preprocess dataset
dataset_path = 'glaucoma_dataset.csv'
df = load_dataset(dataset_path)
df = preprocess_dataset(df)

# Define features and target
feature_cols = ['Age', 'Gender', 'IOP', 'Age_IOP', 'Age_squared', 'IOP_squared', 'log_Age', 'log_IOP']
X = df[feature_cols]
y = df['Diagnosis']

# Use SMOTETomek for more balanced yet realistic dataset
smote_tomek = SMOTETomek(random_state=42)
X, y = smote_tomek.fit_resample(X, y)

# Split dataset with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models with optimized hyperparameters
models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=800, 
        max_depth=30, 
        min_samples_split=2, 
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        class_weight='balanced',
        random_state=42
    ),
    'SVM': SVC(
        kernel='rbf', 
        C=500, 
        gamma='auto', 
        probability=True, 
        class_weight='balanced',
        random_state=42
    ),
    'LogisticRegression': LogisticRegression(
        C=10, 
        penalty='l2',
        solver='liblinear',
        class_weight='balanced',
        max_iter=5000, 
        random_state=42
    ),
    'KNN': KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',
        metric='minkowski',
        p=1,
        n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=500, 
        learning_rate=0.01, 
        max_depth=15, 
        min_samples_split=2,
        min_samples_leaf=2,
        subsample=0.8,
        max_features=0.8,
        random_state=42
    )
}

results = {}

print("Model Performance Evaluation:")
print("-" * 50)

for model_name, model in models.items():
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on training and test sets
    train_accuracy = model.score(X_train_scaled, y_train)
    test_accuracy = model.score(X_test_scaled, y_test)
    
    # Store results
    results[model_name] = {'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy}
    
    # Print detailed results
    y_pred = model.predict(X_test_scaled)
    print(f"\n{model_name} Model:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(classification_report(y_test, y_pred))

# Find best performing model
best_model_name = max(results, key=lambda x: results[x]['test_accuracy'])
best_model = models[best_model_name]

print(f"\nBest Model: {best_model_name}")
print(f"Training Accuracy: {results[best_model_name]['train_accuracy']:.4f}")
print(f"Test Accuracy: {results[best_model_name]['test_accuracy']:.4f}")

# Save best model
with open(f'{best_model_name}_glaucoma_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Evaluate with cross-validation
print("\nCross-Validation Evaluation:")
cross_val_scores = cross_val_score(best_model, X, y, cv=5)
print(f"Cross-Validation Scores: {cross_val_scores}")
print(f"Average Cross-Validation Score: {np.mean(cross_val_scores):.4f}")

# Further tune best model with grid search if it's RandomForest
if best_model_name == "RandomForest":
    print("\nPerforming Grid Search for RandomForest:")
    param_grid = {
        'n_estimators': [400, 500, 600],
        'max_depth': [15, 20, 25],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'), 
                              param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Grid Search Accuracy: {grid_search.best_score_:.4f}")
    
    # Update model with best parameters
    best_model = grid_search.best_estimator_
    best_model_accuracy = best_model.score(X_test_scaled, y_test)
    print(f"Updated Test Accuracy: {best_model_accuracy:.4f}")
    
    # Save improved model
    with open(f'{best_model_name}_optimized_glaucoma_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

# For sensor data prediction
try:
    sensor_data_path = 'sensor_data.csv'
    sensor_data = load_dataset(sensor_data_path)
    
    # Prepare sensor data with the same features as training data
    sensor_data_processed = sensor_data[['Piezo', 'FSR', 'IOP']]
    
    # Add demographic data (sample values)
    age = 25  
    gender = 0  
    
    # Create base features
    sensor_data_processed['Age'] = age
    sensor_data_processed['Gender'] = gender
    
    # Create derived features (same as in training)
    sensor_data_processed['Age_IOP'] = sensor_data_processed['Age'] * sensor_data_processed['IOP']
    sensor_data_processed['Age_squared'] = sensor_data_processed['Age'] ** 2
    sensor_data_processed['IOP_squared'] = sensor_data_processed['IOP'] ** 2
    sensor_data_processed['log_Age'] = np.log1p(sensor_data_processed['Age'])
    sensor_data_processed['log_IOP'] = np.log1p(sensor_data_processed['IOP'])
    
    # Select same features as training
    sensor_data_processed = sensor_data_processed[feature_cols]
    
    # Scale using the same scaler
    sensor_data_scaled = scaler.transform(sensor_data_processed)
    
    # Make predictions
    glaucoma_predictions = best_model.predict(sensor_data_scaled)
    probability_predictions = best_model.predict_proba(sensor_data_scaled)
    
    # Get majority vote
    unique, counts = np.unique(glaucoma_predictions, return_counts=True)
    majority_vote = unique[np.argmax(counts)]
    
    # Calculate confidence
    avg_probability = np.mean(probability_predictions, axis=0)[majority_vote]
    
    # Output prediction
    if majority_vote == 1:
        print(f"\nFinal Prediction: Glaucoma (Confidence: {avg_probability:.2f})")
    else:
        print(f"\nFinal Prediction: No Glaucoma (Confidence: {avg_probability:.2f})")

except Exception as e:
    print(f"Error in sensor data processing: {e}")