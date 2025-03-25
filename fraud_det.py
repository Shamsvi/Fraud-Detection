import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Load data
df = pd.read_csv("https://github.com/Shamsvi/Fraud-Detection/raw/187a24ff26414efc1aabc9e40201ae3bac6b606a/Fraud%20Detection%20Dataset.csv")

# Encode categorical columns
df_encoded = pd.get_dummies(df, drop_first=True)

# Prepare features and target
X = df_encoded.drop(columns=['Fraudulent'], errors='ignore')
y = df_encoded['Fraudulent']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
log_reg = LogisticRegression(class_weight='balanced', random_state=42)
log_reg.fit(X_train_scaled, y_train)

rf = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

xgb_model = XGBClassifier(scale_pos_weight=(y.value_counts()[0] / y.value_counts()[1]), eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

models = {
    'Logistic Regression': (log_reg, X_test_scaled),
    'Random Forest': (rf, X_test),
    'XGBoost': (xgb_model, X_test)
}

# Streamlit Dashboard
st.title("Fraud Detection Model Dashboard")

model_choice = st.selectbox("Select Model", list(models.keys()))
model, test_data = models[model_choice]

# Predictions
y_pred = model.predict(test_data)
y_proba = model.predict_proba(test_data)[:, 1] if hasattr(model, "predict_proba") else None

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
st.subheader("Confusion Matrix")
fig_cm, ax_cm = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Fraud", "Fraud"], yticklabels=["Not Fraud", "Fraud"], ax=ax_cm)
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig_cm)

# Classification Report
st.subheader("Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# ROC Curve
if y_proba is not None:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    st.subheader("ROC Curve")
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend()
    st.pyplot(fig_roc)

# Feature Importance
st.subheader("Feature Importance")
if hasattr(model, 'feature_importances_'):
    importance = model.feature_importances_
    feature_imp_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_imp_df, ax=ax_imp, palette="viridis")
    st.pyplot(fig_imp)
elif model_choice == 'Logistic Regression':
    coef = model.coef_[0]
    feature_imp_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": coef
    }).sort_values(by="Coefficient", key=abs, ascending=False)

    fig_coef, ax_coef = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Coefficient", y="Feature", data=feature_imp_df, ax=ax_coef, palette="coolwarm")
    st.pyplot(fig_coef)
else:
    st.write("Feature importance not available for this model.")