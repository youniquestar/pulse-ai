# import pandas as pd
# import seaborn as sns
# import xgboost as xgb
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix
# from imblearn.over_sampling import SMOTE

# load the data
# df = pd.read_csv('data/ai4i2020.csv')

# Drop non-numeric identifiers (like Product ID)
# df_numeric = df.drop(columns=['UDI', 'Product ID', 'Type'])

# plot feature correlation matrix
# plt.figure(figsize=(10, 8))
# sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm')
# plt.title('Feature Correlation Matrix: Identifying Failure Drivers')
# plt.show()


# COMMENT: VISUALIZING THE INITIAL CORRELATION MATRIX TO SEE HOW THINGS LOOK LIKE IN THE AI4I2020 DATASET. I WILL NOW MOVE ON TO FEATURE SCALING.


# # feature engineering (physics/domain knowledge)
# df['Temp_Delta'] = df['Process temperature [K]'] - df['Air temperature [K]']
# df['Power'] = df['Rotational speed [rpm]'] * df['Torque [Nm]']
#
# # one-hot encoding for the 'Type' column (L, M, H)
# df = pd.get_dummies(df, columns=['Type'], drop_first=True)
#
# # defining the features (X) and target (y) while dropping arbitrary labels.
# X = df.drop(columns=['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
# y = df['Machine failure']
#
# # train/test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#
# # feature scaling (standardization)
# scaler = StandardScaler()
#
# # fit the scaler on the training data, then transform both
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)


# COMMENT: THIS IS A VERY HEALTHY DATA SO I'M GOING TO TRY TO CREATE A BASELINE MODEL, THEN TEST OUT IT'S PRECISION AND SENSITIVITY.


# # train the baseline model
# baseline_model = RandomForestClassifier(random_state=42)
# baseline_model.fit(X_train_scaled, y_train)
#
# # make predictions on the test set
# y_pred = baseline_model.predict(X_test_scaled)

# # evaluate the base model
# print(classification_report(y_test, y_pred))
#
# # plot baseline confusion matrix
# plt.figure(figsize=(6, 4))
# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.ylabel('Actual Label (0 = Healthy, 1 = Failure)')
# plt.xlabel('Predicted Label')
# plt.title('Baseline Confusion Matrix')
# plt.show()


# COMMENT: PRECISION-0.93, RECALL-0.79. I WILL TRY TO CHANGE THE INTERNAL REWARD SYSTEM TO PLACE MORE EMPHASIS ON REDUCING MISSED FAILURES, BY CHANGING THE CLASS WEIGHTS.


# initialize the random forests with custom class weights
# weighted_model = RandomForestClassifier(class_weight={0: 1, 1: 25}, random_state=42)
# weighted_model.fit(X_train_scaled, y_train)
#
# # predict on the test set
# y_pred_weighted = weighted_model.predict(X_test_scaled)
#
# # evaluate the weighted model
# print(classification_report(y_test, y_pred_weighted))

# plot baseline confusion matrix
# plt.figure(figsize=(6, 4))
# cm_weighted = confusion_matrix(y_test, y_pred_weighted)
# sns.heatmap(cm_weighted, annot=True, fmt='d', cmap='Oranges')
# plt.ylabel('Actual Label (0 = Healthy, 1 = Failure)')
# plt.xlabel('Predicted Label')
# plt.title('Weighted Confusion Matrix')
# plt.show()


# COMMENT: PRECISION-0.94, RECALL-0.71. THE MODEL STRUGGLED TO LEARN FROM THE DIFFERENCE IN CLASS WEIGHTS, I'LL BE MOVING ON TO TRYING SMOTE RN ( SYNTHETIC MINORITY OVER-SAMPLING TECHNIQUE)


# smote = SMOTE(random_state=42)
#
# # synthesize the training data
# X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
#
# print(f"Original training dfailures: {sum(y_train == 1)}")
# print(f"Synthesized training failures: {sum(y_train_smote == 1)}")
#
# smote_model = RandomForestClassifier(random_state=42)
# smote_model.fit(X_train_smote, y_train_smote)
#
# # predict on the original test set
# y_pred_smote = smote_model.predict(X_test_scaled)
#
# # result
# print(classification_report(y_test, y_pred_smote))
#
# # plot synthesized confusion matrix
# plt.figure(figsize=(6, 4))
# cm_synthesized = confusion_matrix(y_test, y_pred_smote)
# sns.heatmap(cm_synthesized, annot=True, fmt='d', cmap='Oranges')
# plt.ylabel('Actual Label (0 = Healthy, 1 = Failure)')
# plt.xlabel('Predicted Label')
# plt.title('Synthesized Confusion Matrix')
# plt.show()


# COMMENT: PRECISION: 0.61, RECALL- 0.82. MOVING ON TO XGBOOST.


# # calculate the optimal scale_pos_weight
# healthy_count = sum(y_train == 0)
# failure_count = sum(y_train == 1)
# optimal_weight = healthy_count / failure_count
#
# print(f"Calculated scale_pos_weight: {optimal_weight:.2f}")
#
# # initialize the xgboost classifier
# xgb_base = xgb.XGBClassifier(
#     scale_pos_weight=optimal_weight,
#     min_child_weight=2,
#     n_estimators=100,
#     random_state=42,
#     eval_metric='logloss'
# )

# # train the model on the original training data
# xgb_model.fit(X_train_scaled, y_train)
#
# # predict on the test set
# y_pred_xgb = xgb_model.predict(X_test_scaled)
#
# # classification report
# print(classification_report(y_test, y_pred_xgb))


# COMMENT: PRECISION: 0.51, RECALL: 0.91. MOVING ON TO USING GRIDSEARCHCV TO FIND OUT THE MOST OPTIMAL PARAMETERS


# # define the grid of parameters to test
# param_grid = {
#     'max_depth': [3, 4, 5],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'min_child_weight': [1, 3, 5]
# }
#
# # initialize GridSearchCV targeting F1-Score to balance Precision
# grid_search = GridSearchCV(
#     estimator=xgb_base,
#     param_grid=param_grid,
#     scoring='f1',
#     cv=3,
#     verbose=1,
#     n_jobs=-1
# )
#
# # run on grid search on scaled training data
# grid_search.fit(X_train_scaled, y_train)
#
# # extract the best model configuration
# best_xgb = grid_search.best_estimator_
# print(f"Optimal Parameters Found: {grid_search.best_params_}")
#
# # print the current performance report
# y_pred_grid = best_xgb.predict(X_test_scaled)
# print(classification_report(y_test, y_pred_grid))


# COMMENT: PRECISION: 0.64, RECALL: 0.82. THE BEST PARAMETERS FOR THE XGBOOST MODEL WAS FOUND