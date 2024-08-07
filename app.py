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

# Membuat sidebar untuk navigasi
menu = st.sidebar.selectbox("Menu Options", ["Introduction", "Dataset Selection and Exploration", "Data Preprocessing", "Model Training and Comparison"])

# Fungsi untuk memuat dataset
@st.cache
def load_data():
    data = pd.read_csv("restaurant_menu_optimization_data.csv")
    return data

data = load_data()

# Menu pertama: Deskripsi Data
if menu == "Introduction":
    st.write("### Dataset Predict Restaurant Menu Items Profitability")
    st.write(data)
    # Isi deskripsi sesuai keinginan Anda di sini.
    st.write("### Description")
    st.write("Dataset ini berisi informasi tentang berbagai item menu dari sebuah restoran, termasuk profitabilitasnya. Data ini dikumpulkan untuk menganalisis dan meningkatkan menu restoran dengan mengidentifikasi item-item yang paling menguntungkan dan yang kurang menguntungkan.")

# Menu kedua: Data Info dan Statistik
elif menu == "Dataset Selection and Exploration":
    st.write("### Dataset Predict Restaurant Menu Items Profitability")
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

# Menu ketiga: Stage 1
elif menu == "Data Preprocessing":
    st.write("## Data Preprocessing")
    st.write("### Data sebelum transformasi")
    st.write(data)
    st.write("### Data setelah transformasi")
    label_encoder_menu = LabelEncoder()
    label_encoder_profit = LabelEncoder()
    data['MenuCategory'] = label_encoder_menu.fit_transform(data['MenuCategory'])
    data['Profitability'] = label_encoder_profit.fit_transform(data['Profitability'])

    st.write(data)

    scaler = StandardScaler()
    data[['Price']] = scaler.fit_transform(data[['Price']])

    menu_category_mapping = dict(zip(label_encoder_menu.classes_, label_encoder_menu.transform(label_encoder_menu.classes_)))
    profitability_mapping = dict(zip(label_encoder_profit.classes_, label_encoder_profit.transform(label_encoder_profit.classes_)))

    st.write("### Label Kategori Menu")
    st.json(menu_category_mapping, expanded=True)

    st.write("### Label Profitabilitas")
    st.json(profitability_mapping, expanded=True)
# Menu keempat: Stage 2 dan 3
elif menu == "Model Training and Comparison":
    st.write("## Model Training and Comparison")

    # Memisahkan fitur dan target
    X = data[['Price']]
    y = data['Profitability']

    # Memisahkan data latih dan uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Model selection
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

    # Visualisasi hasil
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

    # Penyesuaian hiperparameter untuk Logistic Regression
    log_param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    }
    log_grid_search = GridSearchCV(LogisticRegression(), log_param_grid, cv=5, scoring='accuracy')
    log_grid_search.fit(X_train, y_train)
    log_best_model = log_grid_search.best_estimator_
    log_preds_best = log_best_model.predict(X_test)

    st.write("### Logistic Regression Best Params")
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

    st.write("### Decision Tree Best Params")
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

    st.write("### SVM Best Params")
    st.write(svm_grid_search.best_params_)
    st.write("SVM Best Accuracy:", accuracy_score(y_test, svm_preds_best))
    st.write("Precision:", precision_score(y_test, svm_preds_best, average='weighted'))
    st.write("Recall:", recall_score(y_test, svm_preds_best, average='weighted'))
    st.write("F1 Score:", f1_score(y_test, svm_preds_best, average='weighted'))

    # Mendefinisikan model dengan parameter terbaik
    log_model = LogisticRegression(C=0.01, solver='liblinear')
    dt_model = DecisionTreeClassifier(max_depth=3, min_samples_split=2, min_samples_leaf=1)
    svm_model = SVC(C=100, gamma=0.01, kernel='rbf')

    # Implementasi cross-validation
    log_scores = cross_val_score(log_best_model, X, y, cv=5, scoring='accuracy')
    dt_scores = cross_val_score(dt_best_model, X, y, cv=5, scoring='accuracy')
    svm_scores = cross_val_score(svm_best_model, X, y, cv=5, scoring='accuracy')

    # Menyiapkan DataFrame untuk hasil
    results = pd.DataFrame({
        'Model': ['Logistic Regression', 'Decision Tree', 'SVM'],
        'Cross-Validation Scores': [list(log_scores), list(dt_scores), list(svm_scores)],
        'Mean CV Accuracy': [log_scores.mean(), dt_scores.mean(), svm_scores.mean()]
    })

    # Menampilkan hasil dengan Streamlit
    st.write("## Model Evaluation with Cross-Validation")

    # Tabel hasil cross-validation
    st.write("### Cross-Validation Results")
    st.write(results)

    # Menyimpan hasil akurasi
    accuracy_scores = {
        'Logistic Regression': log_scores.mean(),
        'Decision Tree': dt_scores.mean(),
        'SVM': svm_scores.mean()
    }

    # Pastikan accuracy_scores tidak kosong sebelum menentukan model terbaik
    if accuracy_scores:
        try:
            # Menentukan model terbaik
            best_model_name = max(accuracy_scores, key=accuracy_scores.get)
            best_model_accuracy = accuracy_scores[best_model_name]

            # Menampilkan model terbaik dengan Streamlit
            st.write("### Best Model")
            st.write(f"Model terbaik adalah {best_model_name} dengan Mean CV Accuracy: {best_model_accuracy:.4f}")
        except Exception as e:
            st.write("### Error")
            st.write(f"Terjadi kesalahan: {e}")
    else:
        st.write("### Best Model")
        st.write("Tidak ada data untuk menentukan model terbaik.")
