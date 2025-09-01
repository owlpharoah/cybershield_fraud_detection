import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def create_features(data):
    """Create engineered features for fraud detection"""
    data = data.copy()
    
    # 1. Balance Change Features  
    data['balance_change_org'] = data['newbalanceOrig'] - data['oldbalanceOrg']  
    data['balance_change_dest'] = data['newbalanceDest'] - data['oldbalanceDest']  
    data['total_balance_change'] = data['balance_change_org'] + data['balance_change_dest']  
      
    # 2. Transaction Pattern Features  
    data['is_zero_balance_after'] = ((data['newbalanceOrig'] == 0) & (data['oldbalanceOrg'] > 0)).astype(int)  
    data['is_full_amount_transfer'] = ((data['amount'] == data['oldbalanceOrg']) & (data['oldbalanceOrg'] > 0)).astype(int)  
    data['is_account_emptied'] = ((data['newbalanceOrig'] == 0) & (data['oldbalanceOrg'] > 1000)).astype(int)  
      
    # 3. Amount Ratio Features  
    data['amount_balance_ratio_org'] = np.where(  
        data['oldbalanceOrg'] > 0,   
        data['amount'] / data['oldbalanceOrg'],   
        0  
    )  
    data['amount_balance_ratio_dest'] = np.where(  
        data['oldbalanceDest'] > 0,   
        data['amount'] / data['oldbalanceDest'],   
        0  
    )  
      
    # 4. Balance Discrepancy Features  
    data['balance_discrepancy_org'] = data['oldbalanceOrg'] - data['amount'] - data['newbalanceOrig']  
    data['balance_discrepancy_dest'] = data['oldbalanceDest'] + data['amount'] - data['newbalanceDest']  
    data['has_balance_discrepancy'] = ((abs(data['balance_discrepancy_org']) > 0.01) |   
                                      (abs(data['balance_discrepancy_dest']) > 0.01)).astype(int)  
      
    # 5. Transaction Size Features  
    data['is_large_transaction'] = (data['amount'] > 100000).astype(int)  
    data['is_very_large_transaction'] = (data['amount'] > 500000).astype(int)  
      
    # 6. Account Relationship Features  
    data['is_customer_to_customer'] = ((data['origin_account_type'] == 'C') &   
                                      (data['dest_account_type'] == 'C')).astype(int)  
    data['is_merchant_involved'] = ((data['origin_account_type'] == 'M') |   
                                   (data['dest_account_type'] == 'M')).astype(int)  
      
    # 7. Interaction Features  
    data['transfer_to_new_account'] = ((data['type'] == 'TRANSFER') &   
                                      (data['oldbalanceDest'] == 0)).astype(int)  
    data['cashout_with_discrepancy'] = ((data['type'] == 'CASH_OUT') &   
                                       (data['has_balance_discrepancy'] == 1)).astype(int)  
      
    # 8. Log Transformations  
    data['log_amount'] = np.log1p(data['amount'])  
    data['log_oldbalanceOrg'] = np.log1p(data['oldbalanceOrg'] + 1)  
      
    # 9. Difference Features  
    data['amount_balance_diff'] = data['amount'] - data['oldbalanceOrg']  
      
    # 10. Flag-based Features  
    data['is_high_value_flagged'] = ((data['isFlaggedFraud'] == 1) &   
                                    (data['amount'] > 100000)).astype(int)  
      
    return data

def generate_synthetic_data(n_samples=10000):
    """Generate synthetic fraud detection data for demonstration"""
    np.random.seed(42)
    
    # Generate base transaction data
    data = {
        'type': np.random.choice(['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'], n_samples),
        'amount': np.random.exponential(1000, n_samples),
        'oldbalanceOrg': np.random.exponential(5000, n_samples),
        'newbalanceOrig': np.zeros(n_samples),
        'oldbalanceDest': np.random.exponential(3000, n_samples),
        'newbalanceDest': np.zeros(n_samples),
        'isFlaggedFraud': np.zeros(n_samples),
        'origin_account_type': np.random.choice(['C', 'M'], n_samples),
        'dest_account_type': np.random.choice(['C', 'M'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Calculate realistic new balances
    df['newbalanceOrig'] = df['oldbalanceOrg'] - df['amount']
    df['newbalanceDest'] = df['oldbalanceDest'] + df['amount']
    
    # Ensure non-negative balances
    df['newbalanceOrig'] = df['newbalanceOrig'].clip(lower=0)
    df['newbalanceDest'] = df['newbalanceDest'].clip(lower=0)
    
    # Create fraud labels based on patterns
    fraud_conditions = (
        (df['amount'] > 100000) &  # Large amounts
        (df['type'].isin(['TRANSFER', 'CASH_OUT'])) &  # High-risk types
        (df['newbalanceOrig'] == 0) &  # Account emptied
        (df['oldbalanceOrg'] > 1000)  # Had significant balance
    )
    
    df['isFraud'] = fraud_conditions.astype(int)
    
    # Add some random fraud cases
    random_fraud = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    df['isFraud'] = df['isFraud'] | random_fraud
    
    # Set flagged fraud based on fraud status
    df['isFlaggedFraud'] = (df['isFraud'] & (df['amount'] > 50000)).astype(int)
    
    return df

def train_fraud_detection_model():
    """Train a new fraud detection model with current scikit-learn version"""
    print("🔄 Generating synthetic training data...")
    
    # Generate synthetic data (in production, you'd load your real dataset)
    df = generate_synthetic_data(n_samples=50000)
    
    print(f"📊 Dataset shape: {df.shape}")
    print(f"🎯 Fraud rate: {df['isFraud'].mean():.2%}")
    
    # Apply feature engineering
    print("🔧 Creating engineered features...")
    df_engineered = create_features(df)
    
    # Select features for training
    feature_columns = [
        'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
        'isFlaggedFraud', 'origin_account_type', 'dest_account_type',
        'balance_change_org', 'balance_change_dest', 'total_balance_change',
        'is_zero_balance_after', 'is_full_amount_transfer', 'is_account_emptied',
        'amount_balance_ratio_org', 'amount_balance_ratio_dest',
        'balance_discrepancy_org', 'balance_discrepancy_dest', 'has_balance_discrepancy',
        'is_large_transaction', 'is_very_large_transaction',
        'is_customer_to_customer', 'is_merchant_involved',
        'transfer_to_new_account', 'cashout_with_discrepancy',
        'log_amount', 'log_oldbalanceOrg', 'amount_balance_diff', 'is_high_value_flagged'
    ]
    
    X = df_engineered[feature_columns]
    y = df_engineered['isFraud']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📈 Training set: {X_train.shape[0]} samples")
    print(f"🧪 Test set: {X_test.shape[0]} samples")
    
    # Define preprocessing pipeline
    categorical_features = ['type', 'origin_account_type', 'dest_account_type']
    numerical_features = [col for col in X.columns if col not in categorical_features]
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Create full pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        ))
    ])
    
    # Train the model
    print("🚀 Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate the model
    print("📊 Evaluating model performance...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS")
    print("="*50)
    
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"Accuracy: {(y_pred == y_test).mean():.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save the model
    print("\n💾 Saving model...")
    joblib.dump(model, 'fraud_detection_model_final.pkl')
    print("✅ Model saved successfully as 'fraud_detection_model_final.pkl'")
    
    # Test loading the model
    print("🔍 Testing model loading...")
    try:
        loaded_model = joblib.load('fraud_detection_model_final.pkl')
        test_prediction = loaded_model.predict(X_test[:1])
        print(f"✅ Model loads successfully! Test prediction: {test_prediction[0]}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
    
    return model

if __name__ == "__main__":
    print("🎯 Fraud Detection Model Retraining")
    print("="*40)
    print("This script will retrain the fraud detection model")
    print("with the current scikit-learn version to fix compatibility issues.")
    print()
    
    model = train_fraud_detection_model()
    
    print("\n🎉 Retraining completed successfully!")
    print("You can now run your Streamlit app without compatibility issues.")
