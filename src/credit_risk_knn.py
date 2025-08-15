# credit_risk_knn.py - Enhanced Version

# 1. Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import openml
import warnings
warnings.filterwarnings('ignore')

def load_german_credit_data():
    """Load German Credit dataset from OpenML with error handling"""
    try:
        print("Downloading dataset...")
        dataset = openml.datasets.get_dataset(31)  # German Credit dataset
        X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)
        df = pd.DataFrame(X, columns=attribute_names)
        df['Class'] = y
        print("Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
        return df, categorical_indicator
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

def preprocess_data(df, categorical_indicator):
    """Comprehensive data preprocessing"""
    # 3. Check for missing values
    print("\nMissing values before imputation:")
    print(df.isna().sum())
    
    # 4. Handle missing values
    # Identify categorical and numeric columns
    categorical_cols = [col for col, cat in zip(df.columns[:-1], categorical_indicator) if cat]
    numeric_cols = [col for col in df.columns if col not in categorical_cols + ['Class']]
    
    print(f"\nCategorical columns: {categorical_cols}")
    print(f"Numeric columns: {numeric_cols}")
    
    # Create copies to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Impute categorical features with most frequent value
    if categorical_cols:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])
    
    # Impute numeric features with median
    if numeric_cols:
        imputer_num = SimpleImputer(strategy='median')
        df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])
    
    # Verify missing values fixed
    print("\nMissing values after imputation:")
    print(df.isna().sum())
    
    # 5. Encode categorical features
    label_encoders = {}
    for col in categorical_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))
        label_encoders[col] = encoder
    
    # Encode target variable if it's categorical
    target_encoder = None
    if df['Class'].dtype == 'object':
        target_encoder = LabelEncoder()
        df['Class'] = target_encoder.fit_transform(df['Class'])
    
    print(f"\nTarget distribution:")
    print(df['Class'].value_counts())
    
    return df, categorical_cols, numeric_cols, label_encoders, target_encoder

def split_and_scale_data(df, numeric_cols):
    """Split data and scale numeric features"""
    # 6. Split dataset
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # 80% train, 10% validation, 10% test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1111, random_state=42, stratify=y_train_full
    )
    
    print(f"\nDataset split:")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # 7. Scale numeric features
    scaler = StandardScaler()
    if numeric_cols:
        # Make copies to avoid modifying original DataFrames
        X_train = X_train.copy()
        X_val = X_val.copy()
        X_test = X_test.copy()
        
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

def find_best_knn(X_train, y_train, X_val, y_val):
    """Find best k for KNN classifier"""
    best_k = 0
    best_acc = 0
    k_results = []
    
    print("\n--- KNN Classifier Hyperparameter Tuning ---")
    print("k\tValidation Accuracy")
    print("-" * 25)
    
    for k in range(1, 21):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_val_pred = knn.predict(X_val)
        acc = accuracy_score(y_val, y_val_pred)
        k_results.append((k, acc))
        
        print(f"{k}\t{acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_k = k
    
    print("-" * 25)
    print(f"Best k: {best_k} with validation accuracy: {best_acc:.4f}")
    
    return best_k, k_results

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    
    print(f"\n--- {model_name} Results ---")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Confusion Matrix:\n{conf_mat}")
    
    # Calculate additional metrics
    tn, fp, fn, tp = conf_mat.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return acc, y_pred

def main():
    """Main execution function"""
    # Load data
    df, categorical_indicator = load_german_credit_data()
    
    if df is None:
        print("Failed to load dataset. Exiting.")
        return
    
    # Preprocess data
    df_processed, categorical_cols, numeric_cols, label_encoders, target_encoder = preprocess_data(df, categorical_indicator)
    
    # Split and scale data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale_data(df_processed, numeric_cols)
    
    # Find best KNN
    best_k, k_results = find_best_knn(X_train, y_train, X_val, y_val)
    
    # Train best KNN on full training data (train + validation)
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])
    
    knn_best = KNeighborsClassifier(n_neighbors=best_k)
    knn_best.fit(X_train_full, y_train_full)
    
    # Evaluate KNN
    knn_acc, knn_pred = evaluate_model(knn_best, X_test, y_test, "Best KNN")
    
    # Train and evaluate Random Forest
    print("\n--- Training Random Forest ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_full, y_train_full)
    rf_acc, rf_pred = evaluate_model(rf, X_test, y_test, "Random Forest")
    
    # Feature importance for Random Forest
    feature_importance = pd.DataFrame({
        'feature': X_train_full.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Summary comparison
    print(f"\n--- Model Comparison ---")
    print(f"KNN (k={best_k}) Test Accuracy: {knn_acc:.4f}")
    print(f"Random Forest Test Accuracy: {rf_acc:.4f}")
    
    if rf_acc > knn_acc:
        print("Random Forest performs better!")
    elif knn_acc > rf_acc:
        print("KNN performs better!")
    else:
        print("Both models perform equally!")
    
    return {
        'best_knn': knn_best,
        'random_forest': rf,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'target_encoder': target_encoder,
        'k_results': k_results,
        'feature_importance': feature_importance
    }

if __name__ == "__main__":
    results = main()