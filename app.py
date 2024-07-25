import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import io
import json

# Judul aplikasi
st.title('Restaurant Menu Optimization')

# Memuat dataset
@st.cache
def load_data():
    data = pd.read_csv("restaurant_menu_optimization_data.csv")
    return data

data = load_data()
st.write("## Dataset")
data

# Menampilkan deskripsi data
st.write("## Data Info")

buffer = io.StringIO()
data.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

st.write("## Descriptive Statistics")
st.write(data.describe())

# Visualisasi distribusi kategori menu
st.write("## Distribusi Kategori Menu")
fig, ax = plt.subplots()
sns.countplot(x='MenuCategory', data=data, ax=ax)
st.pyplot(fig)

# Visualisasi hubungan antara Harga dan Profitabilitas
st.write("## Harga vs Profitabilitas")
fig, ax = plt.subplots()
sns.boxplot(x='Profitability', y='Price', data=data, ax=ax)
st.pyplot(fig)

# Pra-pemrosesan data
st.write("## pemrosesan data")
label_encoder_menu = LabelEncoder()
label_encoder_profit = LabelEncoder()
data['MenuCategory'] = label_encoder_menu.fit_transform(data['MenuCategory'])
data['Profitability'] = label_encoder_profit.fit_transform(data['Profitability'])

st.write("## Data setelah transformasi")
st.write(data)

scaler = StandardScaler()
data[['Price']] = scaler.fit_transform(data[['Price']])

# Menampilkan peta nilai
menu_category_mapping = dict(zip(label_encoder_menu.classes_, label_encoder_menu.transform(label_encoder_menu.classes_)))
profitability_mapping = dict(zip(label_encoder_profit.classes_, label_encoder_profit.transform(label_encoder_profit.classes_)))

st.write("## Mapping Kategori Menu")
st.json(menu_category_mapping, expanded=True)

st.write("## Mapping Profitabilitas")
st.json(profitability_mapping, expanded=True)

# Memisahkan fitur dan target
X = data[['Price']]
y = data['Profitability']

# Memisahkan data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model selection
st.write("## Model Evaluation")

# Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)
log_acc = accuracy_score(y_test, log_preds)
log_prec = precision_score(y_test, log_preds, average='weighted')
log_rec = recall_score(y_test, log_preds, average='weighted')
log_f1 = f1_score(y_test, log_preds, average='weighted')

# Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_preds)
dt_prec = precision_score(y_test, dt_preds, average='weighted')
dt_rec = recall_score(y_test, dt_preds, average='weighted')
dt_f1 = f1_score(y_test, dt_preds, average='weighted')

# SVM
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)
svm_acc = accuracy_score(y_test, svm_preds)
svm_prec = precision_score(y_test, svm_preds, average='weighted')
svm_rec = recall_score(y_test, svm_preds, average='weighted')
svm_f1 = f1_score(y_test, svm_preds, average='weighted')

# Menampilkan hasil
model_performance = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'SVM'],
    'Accuracy': [log_acc, dt_acc, svm_acc],
    'Precision': [log_prec, dt_prec, svm_prec],
    'Recall': [log_rec, dt_rec, svm_rec],
    'F1 Score': [log_f1, dt_f1, svm_f1]
})

st.write(model_performance)

# Visualisasi hasil
st.write("## Model Performance Comparison")
fig, ax = plt.subplots()
sns.barplot(x='Model', y='Accuracy', data=model_performance, ax=ax)
st.pyplot(fig)

fig, ax = plt.subplots()
sns.barplot(x='Model', y='Precision', data=model_performance, ax=ax)
st.pyplot(fig)

fig, ax = plt.subplots()
sns.barplot(x='Model', y='Recall', data=model_performance, ax=ax)
st.pyplot(fig)

fig, ax = plt.subplots()
sns.barplot(x='Model', y='F1 Score', data=model_performance, ax=ax)
st.pyplot(fig)

# Cross-validation untuk setiap model
log_cv_scores = cross_val_score(log_model, X, y, cv=5, scoring='accuracy')
dt_cv_scores = cross_val_score(dt_model, X, y, cv=5, scoring='accuracy')
svm_cv_scores = cross_val_score(svm_model, X, y, cv=5, scoring='accuracy')

st.write("## Cross-validation Scores")
st.write(f"Logistic Regression CV Accuracy: {log_cv_scores.mean()}")
st.write(f"Decision Tree CV Accuracy: {dt_cv_scores.mean()}")
st.write(f"SVM CV Accuracy: {svm_cv_scores.mean()}")

# Penyesuaian hiperparameter untuk Logistic Regression
log_param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs']
}
log_grid_search = GridSearchCV(LogisticRegression(), log_param_grid, cv=5, scoring='accuracy')
log_grid_search.fit(X_train, y_train)
log_best_model = log_grid_search.best_estimator_
log_preds_best = log_best_model.predict(X_test)

st.write("## Logistic Regression Best Params")
st.write(log_grid_search.best_params_)
st.write("Logistic Regression Best Accuracy:", accuracy_score(y_test, log_preds_best))
st.write("Precision:", precision_score(y_test, log_preds_best, average='weighted'))
st.write("Recall:", recall_score(y_test, log_preds_best, average='weighted'))
st.write("F1 Score:", f1_score(y_test, log_preds_best, average='weighted'))

# Penyesuaian hiperparameter untuk Decision Tree
dt_param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10]
}
dt_grid_search = GridSearchCV(DecisionTreeClassifier(), dt_param_grid, cv=5, scoring='accuracy')
dt_grid_search.fit(X_train, y_train)
dt_best_model = dt_grid_search.best_estimator_
dt_preds_best = dt_best_model.predict(X_test)

st.write("## Decision Tree Best Params")
st.write(dt_grid_search.best_params_)
st.write("Decision Tree Best Accuracy:", accuracy_score(y_test, dt_preds_best))
st.write("Precision:", precision_score(y_test, dt_preds_best, average='weighted'))
st.write("Recall:", recall_score(y_test, dt_preds_best, average='weighted'))
st.write("F1 Score:", f1_score(y_test, dt_preds_best, average='weighted'))

# Penyesuaian hiperparameter untuk SVM
svm_param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly']
}
svm_grid_search = GridSearchCV(SVC(), svm_param_grid, cv=5, scoring='accuracy')
svm_grid_search.fit(X_train, y_train)
svm_best_model = svm_grid_search.best_estimator_
svm_preds_best = svm_best_model.predict(X_test)

st.write("## SVM Best Params")
st.write(svm_grid_search.best_params_)
st.write("SVM Best Accuracy:", accuracy_score(y_test, svm_preds_best))
st.write("Precision:", precision_score(y_test, svm_preds_best, average='weighted'))
st.write("Recall:", recall_score(y_test, svm_preds_best, average='weighted'))
st.write("F1 Score:", f1_score(y_test, svm_preds_best, average='weighted'))
