import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Memuat dataset
data = pd.read_csv("C:/MPML/restaurant_menu_optimization_data.csv")

# Deskripsi dataset
print(data.info())
print(data.describe())

print(data['MenuCategory'].head())

# Visualisasi distribusi kategori menu
sns.countplot(x='MenuCategory', data=data)
plt.title('Distribusi Kategori Menu')
plt.show()

# Visualisasi hubungan antara Harga dan Profitabilitas
sns.boxplot(x='Profitability', y='Price', data=data)
plt.title('Harga vs Profitabilitas')
plt.show()

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

print(data.isnull().sum())

# Encoding variabel kategorikal
label_encoder_menu = LabelEncoder()
label_encoder_profit = LabelEncoder()
data['MenuCategory'] = label_encoder_menu.fit_transform(data['MenuCategory'])
data['Profitability'] = label_encoder_profit.fit_transform(data['Profitability'])
# Menampilkan data yang telah di-encode
print(data.head())

# Menampilkan mapping antara nilai numerik dan kategori asli
menu_category_mapping = dict(zip(label_encoder_menu.classes_, label_encoder_menu.transform(label_encoder_menu.classes_)))
profitability_mapping = dict(zip(label_encoder_profit.classes_, label_encoder_profit.transform(label_encoder_profit.classes_)))

print("MenuCategory Mapping:", menu_category_mapping)
print("Profitability Mapping:", profitability_mapping)

# Penskalaan fitur numerik
scaler = StandardScaler()
data[['Price']] = scaler.fit_transform(data[['Price']])

print(data.head())

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Memisahkan fitur dan target
X = data[['Price']]
y = data['Profitability']

# Memisahkan data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model 1: Regresi Logistik
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)
print('Logistic Regression Accuracy:', accuracy_score(y_test, log_preds))
print('Precision:', precision_score(y_test, log_preds, average='weighted'))
print('Recall:', recall_score(y_test, log_preds, average='weighted'))
print('F1 Score:', f1_score(y_test, log_preds, average='weighted'))

# Model 2: Pohon Keputusan
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)
print('Decision Tree Accuracy:', accuracy_score(y_test, dt_preds))
print('Precision:', precision_score(y_test, dt_preds, average='weighted'))
print('Recall:', recall_score(y_test, dt_preds, average='weighted'))
print('F1 Score:', f1_score(y_test, dt_preds, average='weighted'))

# Model 3: SVM
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)
print('SVM Accuracy:', accuracy_score(y_test, svm_preds))
print('Precision:', precision_score(y_test, svm_preds, average='weighted'))
print('Recall:', recall_score(y_test, svm_preds, average='weighted'))
print('F1 Score:', f1_score(y_test, svm_preds, average='weighted'))