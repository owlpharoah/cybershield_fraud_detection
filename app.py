import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        background-color: #f8fafc;
        border: 2px solid #64748b;
    }
    .info-box {
        background-color: #f0f9ff;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Feature engineering function (without step and name columns)
def create_features(data):
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

# Load model function
@st.cache_resource
def load_model():
    try:
        model = joblib.load("fraud_detection_model_final.pkl")
        return model
    except FileNotFoundError:
        st.error("Model file 'fraud_detection_model_final.pkl' not found. Please ensure the model is trained and saved.")
        return None

# Main app
def main():
    st.markdown('<h1 class="main-header">🔍 Financial Fraud Detection System</h1>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar for input
    st.sidebar.header("Transaction Details")
    st.sidebar.markdown("Enter the transaction information below:")
    
    # Input fields
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transaction Information")
        
        transaction_type = st.selectbox(
            "Transaction Type",
            ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"],
            help="Select the type of transaction"
        )
        
        amount = st.number_input(
            "Transaction Amount ($)",
            min_value=0.01,
            value=1000.0,
            step=0.01,
            help="Enter the transaction amount in USD"
        )
        
        is_flagged_fraud = st.selectbox(
            "Is Flagged as Suspicious?",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Whether the transaction was flagged by the system as suspicious"
        )
    
    with col2:
        st.subheader("Account Information")
        
        origin_account_type = st.selectbox(
            "Origin Account Type",
            ["C", "M"],
            format_func=lambda x: "Customer" if x == "C" else "Merchant",
            help="Type of the originating account"
        )
        
        dest_account_type = st.selectbox(
            "Destination Account Type",
            ["C", "M"],
            format_func=lambda x: "Customer" if x == "C" else "Merchant",
            help="Type of the destination account"
        )
    
    # Balance information
    st.subheader("Balance Information")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("**Origin Account Balances**")
        old_balance_org = st.number_input(
            "Old Balance (Origin) ($)",
            min_value=0.0,
            value=10000.0,
            step=0.01,
            help="Balance before the transaction"
        )
        
        new_balance_orig = st.number_input(
            "New Balance (Origin) ($)",
            min_value=0.0,
            value=9000.0,
            step=0.01,
            help="Balance after the transaction"
        )
    
    with col4:
        st.markdown("**Destination Account Balances**")
        old_balance_dest = st.number_input(
            "Old Balance (Destination) ($)",
            min_value=0.0,
            value=5000.0,
            step=0.01,
            help="Balance before receiving the transaction"
        )
        
        new_balance_dest = st.number_input(
            "New Balance (Destination) ($)",
            min_value=0.0,
            value=6000.0,
            step=0.01,
            help="Balance after receiving the transaction"
        )
    
    # Prediction button
    if st.button("🔍 Analyze Transaction", type="primary", use_container_width=True):
        
        # Create input data
        input_data = {
            'type': [transaction_type],
            'amount': [amount],
            'oldbalanceOrg': [old_balance_org],
            'newbalanceOrig': [new_balance_orig],
            'oldbalanceDest': [old_balance_dest],
            'newbalanceDest': [new_balance_dest],
            'isFlaggedFraud': [is_flagged_fraud],
            'origin_account_type': [origin_account_type],
            'dest_account_type': [dest_account_type]
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame(input_data)
        
        # Apply feature engineering
        try:
            input_df = create_features(input_df)
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0, 1]
            
            # Display results
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            st.markdown(f'''
            <div class="prediction-box">
                <h3>Fraud Probability: {probability:.2%}</h3>
            </div>
            ''', unsafe_allow_html=True)
            
            # Additional insights
            st.subheader("📊 Transaction Analysis")
            
            col5, col6, col7 = st.columns(3)
            
            with col5:
                st.metric(
                    "Risk Level",
                    "High" if probability > 0.7 else "Medium" if probability > 0.3 else "Low",
                    f"{probability:.1%}"
                )
            
            with col6:
                balance_change = new_balance_orig - old_balance_org
                st.metric(
                    "Origin Balance Change",
                    f"${balance_change:,.2f}",
                    "Account Emptied" if new_balance_orig == 0 and old_balance_org > 0 else "Normal"
                )
            
            with col7:
                amount_ratio = amount / old_balance_org if old_balance_org > 0 else 0
                st.metric(
                    "Amount/Balance Ratio",
                    f"{amount_ratio:.2%}",
                    "Full Transfer" if amount == old_balance_org and old_balance_org > 0 else "Partial"
                )
            
            # Risk factors
            st.subheader("🚩 Risk Factors Detected")
            
            risk_factors = []
            
            # Check various risk patterns
            if new_balance_orig == 0 and old_balance_org > 0:
                risk_factors.append("Account completely emptied")
            
            if amount == old_balance_org and old_balance_org > 0:
                risk_factors.append("Full balance transfer")
            
            if amount > 100000:
                risk_factors.append("Large transaction amount")
            
            if old_balance_dest == 0 and transaction_type == "TRANSFER":
                risk_factors.append("Transfer to new/empty account")
            
            if abs((old_balance_org - amount) - new_balance_orig) > 0.01:
                risk_factors.append("Balance discrepancy detected")
            
            if transaction_type in ["TRANSFER", "CASH_OUT"]:
                risk_factors.append("High-risk transaction type")
            
            if is_flagged_fraud == 1:
                risk_factors.append("Previously flagged as suspicious")
            
            if risk_factors:
                for factor in risk_factors:
                    st.warning(f"• {factor}")
            else:
                st.info("• No significant risk factors detected")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
    
    # Information section
    st.markdown("---")
    with st.expander("ℹ️ About This System"):
        st.markdown("""
        <div class="info-box">
        <h4>How it works:</h4>
        <p>This fraud detection system uses a LightGBM machine learning model trained on financial transaction data. 
        The model analyzes various features including:</p>
        <ul>
            <li><strong>Transaction patterns:</strong> Amount, type, and timing</li>
            <li><strong>Balance changes:</strong> How account balances change</li>
            <li><strong>Account relationships:</strong> Customer-to-customer vs merchant transactions</li>
            <li><strong>Risk indicators:</strong> Large amounts, account emptying, balance discrepancies</li>
        </ul>
        
        <h4>Input Requirements:</h4>
        <ul>
            <li><strong>Transaction Type:</strong> PAYMENT, TRANSFER, CASH_OUT, DEBIT, or CASH_IN</li>
            <li><strong>Amount:</strong> Transaction amount in USD</li>
            <li><strong>Account Types:</strong> Customer (C) or Merchant (M)</li>
            <li><strong>Balances:</strong> Before and after transaction balances for both accounts</li>
            <li><strong>Flag Status:</strong> Whether transaction was previously flagged</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Sample transaction examples
    with st.expander("📝 Try Sample Transactions"):
        col_sample1, col_sample2 = st.columns(2)
        
        with col_sample1:
            if st.button("🚨 Suspicious Transaction Example"):
                st.session_state.update({
                    'sample_type': 'TRANSFER',
                    'sample_amount': 6311409.28,
                    'sample_old_org': 6311409.28,
                    'sample_new_org': 0.00,
                    'sample_old_dest': 0.00,
                    'sample_new_dest': 0.00,
                    'sample_flagged': 0,
                    'sample_origin_type': 'C',
                    'sample_dest_type': 'C'
                })
                st.rerun()
        
        with col_sample2:
            if st.button("✅ Normal Transaction Example"):
                st.session_state.update({
                    'sample_type': 'PAYMENT',
                    'sample_amount': 150.00,
                    'sample_old_org': 5000.00,
                    'sample_new_org': 4850.00,
                    'sample_old_dest': 1000.00,
                    'sample_new_dest': 1150.00,
                    'sample_flagged': 0,
                    'sample_origin_type': 'C',
                    'sample_dest_type': 'M'
                })
                st.rerun()

    # Model information sidebar
    with st.sidebar:
        st.markdown("---")
        st.subheader("📈 Model Information")
        
        if model is not None:
            st.success("✅ Model loaded successfully")
            
            # Try to get model info
            try:
                st.info(f"Model Type: LightGBM Classifier")
                st.info(f"Features: Engineered from transaction data")
                st.info(f"Preprocessing: StandardScaler + OneHotEncoder")
            except:
                pass
        else:
            st.error("❌ Model not loaded")
        
        st.markdown("---")
        st.subheader("🎯 Quick Tips")
        st.markdown("""
        **High-risk patterns:**
        - Large amounts (>$100K)
        - Account emptying
        - Transfers to new accounts
        - Balance discrepancies
        - CASH_OUT/TRANSFER types
        
        **Low-risk patterns:**
        - Small payments
        - Merchant transactions
        - Consistent balances
        - PAYMENT/DEBIT types
        """)

if __name__ == "__main__":
    main()