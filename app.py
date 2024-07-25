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

# Judul aplikasi
st.title('Restaurant Menu Optimization')

# Memuat dataset
@st.cache
def load_data():
    data = pd.read_csv("restaurant_menu_optimization_data.csv")
    return data

data = load_data()

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
label_encoder_menu = LabelEncoder()
label_encoder_profit = LabelEncoder()
data['MenuCategory'] = label_encoder_menu.fit_transform(data['MenuCategory'])
data['Profitability'] = label_encoder_profit.fit_transform(data['Profitability'])

scaler = StandardScaler()
data[['Price']] = scaler.fit_transform(data[['Price']])

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

# Menampilkan peta nilai
menu_category_mapping = dict(zip(label_encoder_menu.classes_, label_encoder_menu.transform(label_encoder_menu.classes_)))
profitability_mapping = dict(zip(label_encoder_profit.classes_, label_encoder_profit.transform(label_encoder_profit.classes_)))

st.write("## Mapping Kategori Menu")
st.write(menu_category_mapping)
st.write("## Mapping Profitabilitas")
st.write(profitability_mapping)
