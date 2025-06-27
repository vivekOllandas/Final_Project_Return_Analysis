import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv(r'return_orders\ecommerce_returns_synthetic_data.csv')

# Create binary target column
df['Is_Returned'] = df['Return_Status'].apply(lambda x: 1 if x == 'Returned' else 0)

# Drop columns not needed for modeling (IDs, dates, reasons)
drop_cols = ['Order_ID', 'Product_ID', 'User_ID', 'Order_Date', 'Return_Date', 'Return_Reason']
df = df.drop(columns=drop_cols)

# Optional: Reset index
df = df.reset_index(drop=True)

# Select features for modeling
features = ['Product_Category', 'Product_Price', 'Order_Quantity', 'User_Age', 'User_Gender',
            'User_Location', 'Payment_Method', 'Shipping_Method', 'Discount_Applied']
X = df[features]
y = df['Is_Returned']

# One-hot encode categorical features
categorical_cols = ['Product_Category', 'User_Gender', 'User_Location', 'Payment_Method', 'Shipping_Method']
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Scale numeric features
numeric_cols = ['Product_Price', 'Order_Quantity', 'User_Age', 'Discount_Applied']
scaler = StandardScaler()
X_encoded[numeric_cols] = scaler.fit_transform(X_encoded[numeric_cols])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train logistic regression with more iterations
model = LogisticRegression(max_iter=3000)
model.fit(X_train, y_train)

# Predict probabilities
df['Return_Probability'] = model.predict_proba(X_encoded)[:, 1]

# Show top 5 products with highest return probability
print("\nTop 5 high-risk products:")
print(df[['Product_Category', 'Product_Price', 'Order_Quantity', 'Return_Probability']].sort_values('Return_Probability', ascending=False).head())

# Export high-risk products (probability > 0.6) to CSV
high_risk = df[df['Return_Probability'] > 0.6]
high_risk.to_csv('high_risk_products.csv', index=False)

print(f"\nExported {len(high_risk)} high-risk products to high_risk_products.csv")