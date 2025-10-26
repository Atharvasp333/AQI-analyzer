import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import joblib

def prepare_transaction_data(df):
    """Prepare data for association rule mining"""
    # Create categorical bins for continuous variables
    df_rules = df.copy()
    
    # Bin pollutants into categories
    df_rules['PM2.5_Level'] = pd.cut(df_rules['PM2.5'], bins=3, labels=['Low_PM2.5', 'Medium_PM2.5', 'High_PM2.5'])
    df_rules['PM10_Level'] = pd.cut(df_rules['PM10'], bins=3, labels=['Low_PM10', 'Medium_PM10', 'High_PM10'])
    df_rules['NO2_Level'] = pd.cut(df_rules['NO2'], bins=3, labels=['Low_NO2', 'Medium_NO2', 'High_NO2'])
    df_rules['CO_Level'] = pd.cut(df_rules['CO'], bins=3, labels=['Low_CO', 'Medium_CO', 'High_CO'])
    df_rules['Traffic_Level'] = pd.cut(df_rules['Traffic_Volume'], bins=3, labels=['Low_Traffic', 'Medium_Traffic', 'High_Traffic'])
    
    # Create binary matrix for association rules
    categorical_cols = ['PM2.5_Level', 'PM10_Level', 'NO2_Level', 'CO_Level', 'Traffic_Level', 'AQI_Category']
    
    # Convert to binary matrix
    binary_data = []
    for _, row in df_rules.iterrows():
        transaction = []
        for col in categorical_cols:
            if pd.notna(row[col]):
                transaction.append(str(row[col]))
        binary_data.append(transaction)
    
    # Use TransactionEncoder to create binary matrix
    te = TransactionEncoder()
    te_ary = te.fit(binary_data).transform(binary_data)
    df_binary = pd.DataFrame(te_ary, columns=te.columns_)
    
    return df_binary

def mine_association_rules():
    """Mine association rules from air quality data"""
    print("Loading processed data for association rule mining...")
    df = pd.read_csv('data/processed_air_quality_data.csv')
    
    # Prepare transaction data
    df_binary = prepare_transaction_data(df)
    
    # Find frequent itemsets
    print("Finding frequent itemsets...")
    frequent_itemsets = apriori(df_binary, min_support=0.1, use_colnames=True)
    
    if len(frequent_itemsets) == 0:
        print("No frequent itemsets found. Lowering minimum support...")
        frequent_itemsets = apriori(df_binary, min_support=0.05, use_colnames=True)
    
    # Generate association rules
    print("Generating association rules...")
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
    
    if len(rules) == 0:
        print("No rules found with high confidence. Lowering threshold...")
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
    
    # Sort by lift and get top 5
    rules_sorted = rules.sort_values('lift', ascending=False).head(5)
    
    # Format rules for display
    formatted_rules = []
    for _, rule in rules_sorted.iterrows():
        antecedent = ', '.join(list(rule['antecedents']))
        consequent = ', '.join(list(rule['consequents']))
        
        formatted_rule = {
            'rule': f"{antecedent} → {consequent}",
            'support': round(rule['support'], 3),
            'confidence': round(rule['confidence'], 3),
            'lift': round(rule['lift'], 3)
        }
        formatted_rules.append(formatted_rule)
    
    # Save rules
    joblib.dump(formatted_rules, 'models/association_rules.pkl')
    
    print(f"Found {len(formatted_rules)} association rules")
    for rule in formatted_rules:
        print(f"Rule: {rule['rule']}")
        print(f"Support: {rule['support']}, Confidence: {rule['confidence']}, Lift: {rule['lift']}")
        print("-" * 50)
    
    return formatted_rules

if __name__ == "__main__":
    rules = mine_association_rules()
    print("Association rule mining completed!")