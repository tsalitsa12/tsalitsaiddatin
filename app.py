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

# Menu navigasi
menu = st.sidebar.selectbox(
    "Pilih Menu",
    ["Deskripsi Data", "Stage 1", "Stage 2", "Stage 3"]
)

# Memuat dataset
@st.cache
def load_data():
    data = pd.read_csv("restaurant_menu_optimization_data.csv")
    return data

data = load_data()

if menu == "Deskripsi Data":
    st.write("## Deskripsi Data")
    # Memasukkan deskripsi data
    st.write("Masukkan deskripsi dataset di sini.")

elif menu == "Stage 1":
    st.write("## Stage 1")
    st.write("### Dataset")
    st.write(data)

    st.write("### Data Info")
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.write("### Descriptive Statistics")
    st.write(data.describe())

    st.write("### Distribusi Kategori Menu")
    fig, ax = plt.subplots()
    sns.countplot(x='MenuCategory', data=data, ax=ax)
    st.pyplot(fig)

    st.write("### Harga vs Profitabilitas")
    fig, ax = plt.subplots()
    sns.boxplot(x='Profitability', y='Price', data=data, ax=ax)
    st.pyplot(fig)

elif menu == "Stage 2":
    st.write("## Stage 2")
    
    label_encoder_menu = LabelEncoder()
    label_encoder_profit = LabelEncoder()
    data['MenuCategory'] = label_encoder_menu.fit_transform(data['MenuCategory'])
    data['Profitability'] = label_encoder_profit.fit_transform(data['Profitability'])

    st.write("### Data setelah transformasi")
    st.write(data)

    scaler = StandardScaler()
    data[['Price']] = scaler.fit_transform(data[['Price']])

    # Menampilkan peta nilai
    menu_category_mapping = dict(zip(label_encoder_menu.classes_, label_encoder_menu.transform(label_encoder_menu.classes_)))
    profitability_mapping = dict(zip(label_encoder_profit.classes_, label_encoder_profit.transform(label_encoder_profit.classes_)))

    st.write("### Mapping Kategori Menu")
    st.json(menu_category_mapping, expanded=True)

    st.write("### Mapping Profitabilitas")
    st.json(profitability_mapping, expanded=True)

    # Memisahkan fitur dan target
    X = data[['Price']]
    y = data['Profitability']

    # Memisahkan data latih dan uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    st.write("### Model Evaluation")
    
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

    st.write("### Model Performance Comparison")
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

elif menu == "Stage 3":
    st.write("## Stage 3")

    # Penyesuaian hiperparameter untuk Logistic Regression
    log_param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    }
    log_grid_search = GridSearchCV(LogisticRegression(), log_param_grid, cv=5, scoring='accuracy')

    # Tambahkan pengecekan sebelum fitting
    st.write("Tipe Data X_train:", X_train.dtypes)
    st.write("Tipe Data y_train:", y_train.dtypes)
    st.write("Apakah ada nilai hilang di X_train?", X_train.isnull().sum())
    st.write("Apakah ada nilai hilang di y_train?", y_train.isnull().sum())
    st.write("Ukuran X_train:", X_train.shape)
    st.write("Ukuran y_train:", y_train.shape)

    try:
        log_grid_search.fit(X_train, y_train)
    except Exception as e:
        st.write("Terjadi kesalahan saat fitting Logistic Regression GridSearchCV:")
        st.write(e)
        st.stop()

    log_best_model = log_grid_search.best_estimator_
    log_preds_best = log_best_model.predict(X_test)

    st.write("### Logistic Regression Best Params")
    st.write(log_grid_search.best_params_)
    st.write("Logistic Regression Best Accuracy:", accuracy_score(y_test, log_preds_best))
    st.write("Precision:", precision_score(y_test, log_preds_best, average='weighted'))
    st.write("Recall:", recall_score(y_test, log_preds_best, average='weighted'))
    st.write("F1 Score:", f1_score(y_test, log_preds_best, average='weighted'))
