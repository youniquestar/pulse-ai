import joblib
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load the data
df = pd.read_csv('data/ai4i2020.csv')

# feature engineering (physics/domain knowledge)
df['Temp_Delta'] = df['Process temperature [K]'] - df['Air temperature [K]']
df['Power'] = df['Rotational speed [rpm]'] * df['Torque [Nm]']

# one-hot encoding for the 'Type' column (L, M, H)
df = pd.get_dummies(df, columns=['Type'], drop_first=True)

# defining the features (X) and target (y) while dropping arbitrary labels.
X = df.drop(columns=['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
y = df['Machine failure']

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# feature scaling (standardization)
scaler = StandardScaler()

# fit the scaler on the training data, then transform both
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# calculate the optimal scale_pos_weight
healthy_count = sum(y_train == 0)
failure_count = sum(y_train == 1)
optimal_weight = healthy_count / failure_count

print(f"Calculated scale_pos_weight: {optimal_weight:.2f}")

# initialize the xgboost classifier
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=optimal_weight,
    min_child_weight=1,
    learning_rate=0.1,
    max_depth=5,
    n_estimators=100,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# train the model on the original training data
xgb_model.fit(X_train_scaled, y_train)

# predict on the test set
y_pred_xgb = xgb_model.predict(X_test_scaled)

# classification report
print(classification_report(y_test, y_pred_xgb))

# save the trained xgboost model and the standard scaler
joblib.dump(xgb_model, 'main_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and Scaler successfully serialized!")