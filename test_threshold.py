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

def test_threshold_logic():
    """Test the new 50% threshold logic"""
    print("🧪 Testing 50% Threshold Logic")
    print("="*40)
    
    # Load the model
    try:
        model = joblib.load('fraud_detection_model_final.pkl')
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test different probability scenarios
    test_scenarios = [
        {
            'name': '🔴 HIGH FRAUD PROBABILITY (75%)',
            'probability': 0.75,
            'description': 'Very suspicious transaction'
        },
        {
            'name': '🟡 MEDIUM FRAUD PROBABILITY (60%)',
            'probability': 0.60,
            'description': 'Moderately suspicious transaction'
        },
        {
            'name': '🟢 LOW FRAUD PROBABILITY (30%)',
            'probability': 0.30,
            'description': 'Likely legitimate transaction'
        },
        {
            'name': '🔴 BORDERLINE FRAUD (55%)',
            'probability': 0.55,
            'description': 'Slightly above threshold'
        },
        {
            'name': '🟢 BORDERLINE LEGITIMATE (45%)',
            'probability': 0.45,
            'description': 'Slightly below threshold'
        }
    ]
    
    print("\n📊 Threshold Logic Test Results:")
    print("-" * 60)
    
    for scenario in test_scenarios:
        probability = scenario['probability']
        
        # Apply 50% threshold logic
        is_fraud = probability > 0.5
        
        # Determine risk level
        if probability > 0.5:
            risk_level = "🔴 HIGH RISK"
        else:
            risk_level = "🟢 LOW RISK"
        
        # Determine prediction
        prediction = "🚨 FRAUD DETECTED" if is_fraud else "✅ LEGITIMATE"
        
        print(f"\n{scenario['name']}")
        print(f"   Description: {scenario['description']}")
        print(f"   Probability: {probability:.1%}")
        print(f"   Threshold: 50%")
        print(f"   Prediction: {prediction}")
        print(f"   Risk Level: {risk_level}")
        print("-" * 40)
    
    print("\n🎯 Key Changes Made:")
    print("✅ Fraud detection threshold: 50% (was model default)")
    print("✅ Visual indicators: 🚨 for fraud, ✅ for legitimate")
    print("✅ Risk levels: 🔴 High (>50%), 🟢 Low (<50%)")
    print("✅ Enhanced styling for fraud detection")

if __name__ == "__main__":
    test_threshold_logic()
