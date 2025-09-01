import pandas as pd
import numpy as np
import joblib

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

def test_model():
    """Test the model with sample transaction data"""
    print("🧪 Testing Fraud Detection Model")
    print("="*40)
    
    # Load the model
    try:
        model = joblib.load('fraud_detection_model_final.pkl')
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test cases
    test_cases = [
        {
            'name': '🚨 SUSPICIOUS TRANSACTION',
            'data': {
                'type': 'TRANSFER',
                'amount': 6311409.28,
                'oldbalanceOrg': 6311409.28,
                'newbalanceOrig': 0.00,
                'oldbalanceDest': 0.00,
                'newbalanceDest': 0.00,
                'isFlaggedFraud': 0,
                'origin_account_type': 'C',
                'dest_account_type': 'C'
            }
        },
        {
            'name': '✅ NORMAL PAYMENT',
            'data': {
                'type': 'PAYMENT',
                'amount': 150.00,
                'oldbalanceOrg': 5000.00,
                'newbalanceOrig': 4850.00,
                'oldbalanceDest': 1000.00,
                'newbalanceDest': 1150.00,
                'isFlaggedFraud': 0,
                'origin_account_type': 'C',
                'dest_account_type': 'M'
            }
        },
        {
            'name': '⚠️ LARGE CASH OUT',
            'data': {
                'type': 'CASH_OUT',
                'amount': 250000.00,
                'oldbalanceOrg': 300000.00,
                'newbalanceOrig': 50000.00,
                'oldbalanceDest': 0.00,
                'newbalanceDest': 250000.00,
                'isFlaggedFraud': 1,
                'origin_account_type': 'C',
                'dest_account_type': 'C'
            }
        },
        {
            'name': '💰 SMALL TRANSFER',
            'data': {
                'type': 'TRANSFER',
                'amount': 500.00,
                'oldbalanceOrg': 2000.00,
                'newbalanceOrig': 1500.00,
                'oldbalanceDest': 100.00,
                'newbalanceDest': 600.00,
                'isFlaggedFraud': 0,
                'origin_account_type': 'C',
                'dest_account_type': 'C'
            }
        },
        {
            'name': '🚨 HIGH RISK FRAUD',
            'data': {
                'type': 'TRANSFER',
                'amount': 1000000.00,
                'oldbalanceOrg': 1000000.00,
                'newbalanceOrig': 0.00,
                'oldbalanceDest': 0.00,
                'newbalanceDest': 0.00,
                'isFlaggedFraud': 1,
                'origin_account_type': 'C',
                'dest_account_type': 'C'
            }
        }
    ]
    
    print("\n📊 Test Results:")
    print("-" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"   Transaction: {test_case['data']['type']} - ${test_case['data']['amount']:,.2f}")
        
        # Create DataFrame
        input_df = pd.DataFrame([test_case['data']])
        
        # Apply feature engineering
        input_df = create_features(input_df)
        
        # Make prediction
        try:
            probability = model.predict_proba(input_df)[0, 1]
            
            # Use 50% threshold for fraud detection
            is_fraud = probability > 0.5
            
            result = "🚨 FRAUD" if is_fraud else "✅ LEGITIMATE"
            print(f"   Prediction: {result}")
            print(f"   Fraud Probability: {probability:.2%}")
            
            # Risk assessment
            if probability > 0.5:
                risk = "🔴 HIGH RISK"
            else:
                risk = "🟢 LOW RISK"
            print(f"   Risk Level: {risk}")
            
        except Exception as e:
            print(f"   ❌ Prediction error: {e}")
        
        print("-" * 40)
    
    print("\n🎉 Model testing completed!")

if __name__ == "__main__":
    test_model()
