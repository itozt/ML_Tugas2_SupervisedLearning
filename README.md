[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/sPpv8lmn)
# Hands-On 2 Pembelajaran Mesin 2025
| Nama | NRP | Kelas |
|----------|-----------|----------|
| Christoforus Indra Bagus Pratama | 5025231124 | Pembelajaran Mesin - B |

### 1. Inisialisasi Environment
Mengimpor pustaka (numpy, pandas, matplotlib, seaborn, os) dan menonaktifkan peringatan FutureWarning
``` py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
```
### 2. Memuat Data
Membaca file CSV train.csv ke dalam DataFrame df dan menampilkan lima baris pertama.
``` py
df = pd.read_csv('train.csv')
df.head()
```
### 3. Ringkasan Struktur Data
Menampilkan informasi tipe data, jumlah non-null, dan penggunaan memori DataFrame.
``` py
df.info()
```
### 4. Ukuran Data
Mencetak jumlah baris dan kolom pada DataFrame.
``` py
print(f'Number of Lines : {df.shape[0]}')
print(f'Number of Columns : {df.shape[1]}')
```
### 5. Pengecekan Nilai Hilang
Menghitung dan mengurutkan jumlah nilai kosong (NaN) per kolom.
``` py
df.isna().sum().sort_values(ascending=False)
```
### 6. Pengecekan Duplikasi
Menghitung jumlah baris duplikat dalam DataFrame.
``` py
print(f'Duplicate Row : {df.duplicated().sum()}')
```
### 7. Statistik Deskriptif
Menghasilkan ringkasan statistik (mean, std, min, max, dll.) untuk kolom numerik.
``` py
df.describe().T
```
### 8. Jumlah Nilai Unik
Menghitung banyaknya nilai unik per kolom.
``` py
df.nunique()
```
### 9. Identifikasi Fitur Numerik dan Kategorikal
Menentukan num_cols (kolom numerik, kecuali id dan target) dan obj_cols (kolom bertipe objek).
``` py
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.difference(['id', 'will_buy_on_return_visit'])
# Kolom kategorikal (hanya object)
obj_cols = df.select_dtypes(include='object').columns
```
### 10. Duplikasi Seleksi Kolom
Mengulangi penentuan kolom numerik dan kategorikal (duplikat dari Bagian 9).
``` py
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.difference(['id', 'will_buy_on_return_visit'])
# Kolom kategorikal (hanya object)
obj_cols = df.select_dtypes(include='object').columns
```
### 11. Visualisasi Distribusi Kategori
a. Countplot Kategori : Menampilkan plot frekuensi 10 nilai terbanyak setiap kolom kategorikal, dibedakan berdasarkan will_buy_on_return_visit. <br>
b. Histogram Numerik : Menampilkan histogram dan KDE untuk setiap kolom numerik, dibedakan berdasarkan target.
``` py
color_palette = sns.color_palette("Set1", len(obj_cols))

n_cols = 3
n_rows = (len(obj_cols) // n_cols) + 1

plt.figure(figsize=(18, n_rows * 5))

for i, column in enumerate(obj_cols, 1):
    plt.subplot(n_rows, n_cols, i)

    top_10_items = df[column].value_counts().nlargest(10).index

    filtered_df = df[df[column].isin(top_10_items)]

    sns.countplot(data=filtered_df, x=column, palette=color_palette, order=top_10_items, hue='will_buy_on_return_visit')
    plt.title(column)
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```
``` py
n_cols = 3
n_rows = (len(num_cols) // n_cols) + 1

plt.figure(figsize=(18, n_rows * 5))

for i, column in enumerate(num_cols, 1):
    plt.subplot(n_rows, n_cols, i)

    sns.histplot(data=df, x=column, hue='will_buy_on_return_visit', kde=True, palette='Set1', element='step')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
```
### 12. Pairplot Fitur Numerik
Membuat pairplot untuk semua fitur numerik bersama target, memvisualisasikan korelasi dan distribusi bivariate
``` py
combined_df = pd.concat([df[num_cols], df['will_buy_on_return_visit']], axis=1)
sns.pairplot(combined_df, hue='will_buy_on_return_visit', palette='Set1')
plt.show()
```
### 13. Kerangka Feature Engineering
Mendefinisikan fungsi feature_engineer(df) sebagai placeholder untuk transformasi fitur.
``` py
def feature_engineer(df):
    df = df.copy()
    return df
```
### 14. Impor Modul Pembelajaran Mesin
Mengimpor modul scikit-learn (model selection, preprocessing, pipeline), xgboost, dan optuna untuk tuning.
``` py
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
# !pip install optuna
import optuna
```
### 15. Persiapan Data dan Cross-Validation
Menyalin DataFrame, memisahkan X (fitur) dan y (target), serta menyetel StratifiedKFold.
``` py
train = df.copy()
X = train.drop(['id', 'will_buy_on_return_visit'], axis=1)
y = train['will_buy_on_return_visit']
random_state = 42
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
```
### 16. Encoding Kategorikal
Menerapkan LabelEncoder untuk mengubah kolom tipe objek menjadi numerik, menyimpan encoder untuk test set.
``` py
label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
```
### 17. Split Train/Test
Membagi data menjadi X_train, X_test, y_train, dan y_test secara stratified.
``` py
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.14, stratify=y, random_state=random_state)
```
### 18. Evaluasi Model Awal
Mendefinisikan daftar classifier (KNN, GaussianNB, Decision Tree, Random Forest, XGBoost), melakukan cross-validation, dan menampilkan hasil perbandingan.
``` py
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import pandas as pd

# Daftar classifier
classifiers = [
    ("KNN", KNeighborsClassifier(n_neighbors=9)),
    ("GaussianNB", GaussianNB()),
    ("Decision Tree", DecisionTreeClassifier(random_state=random_state)),
    ("Random Forest", RandomForestClassifier(random_state=random_state)),
    ("XGBoost", XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='error')),
]

# Menyimpan hasil cross-validation
cv_means = []
cv_std = []
algorithm_names = []

# Loop untuk cross-validation
for name, clf in classifiers:
    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),  # Lebih universal
        StandardScaler(),
        clf
    )
    scores = cross_val_score(pipeline, X_train, y_train, scoring="accuracy", cv=kfold, n_jobs=4)
    cv_means.append(scores.mean())
    cv_std.append(scores.std())
    algorithm_names.append(name)

# Tabel hasil
cv_res = pd.DataFrame({
    "CrossValMeans": cv_means,
    "CrossValerrors": cv_std,
    "Algorithm": algorithm_names
})

# Urutkan dari yang terbaik
cv_res = cv_res.sort_values(by='CrossValMeans', ascending=False)
print(cv_res)
```
### 19. Definisi Fungsi Objective Optuna
Membuat fungsi objective(trial) untuk mengoptimasi hyperparameter XGBoost via optuna
``` py
def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
        'random_state': random_state,
        'use_label_encoder': False,
        'eval_metric': 'error'
    }

    clf = XGBClassifier(**param)
    score = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1).mean()
    return score
```
### 20. Menjalankan Studi Optuna
Membuat dan menjalankan optuna.create_study selama 1.000 trial untuk memaksimalkan akurasi
``` py
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1000)
```
### 21. Menampilkan Hyperparameter Terbaik
Mengambil dan mencetak best_params hasil tuning Optuna
``` py
best_params = study.best_params
print("Best Hyperparameter :", best_params)
```
### 22. Training dan Evaluasi Final
Melatih XGBClassifier dengan parameter terbaik, memprediksi pada X_test, dan mencetak akurasi validasi.
``` py
clf = XGBClassifier(
    **best_params,
    random_state=random_state,
    use_label_encoder=False,
    eval_metric='error'
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

score = accuracy_score(y_test, y_pred)
print('Val Accuracy:', score)
```
### 23. Memuat dan Menyiapkan Data Test
Membaca test.csv, menerapkan fungsi feature engineering, dan menghapus kolom id.
``` py
test = pd.read_csv('test.csv')
test_df = feature_engineer(test).drop('id', axis=1)

# Assuming test_df and label_encoders are already defined

for col, le in label_encoders.items():
    # Get a set of known values from the LabelEncoder
    known_values = set(le.classes_)

    # Find unknown values in the test data
    unknown_values = set(test_df[col].astype(str)) - known_values

    if unknown_values:
        test_df.loc[test_df[col].astype(str).isin(unknown_values), col] = 'unknown'

    # Add the 'unknown' label to the LabelEncoder's classes if it's not already there
    if 'unknown' not in le.classes_:
        le.classes_ = np.append(le.classes_, 'unknown')

    # Now transform the data, it should handle unknown values gracefully
    test_df[col] = le.transform(test_df[col].astype(str))
```
### 24. Penyelarasan Fitur Test
Mendefinisikan fungsi align_test_features untuk memastikan kolom test sama persis dan berurutan seperti data latih.
``` py
def align_test_features(train_columns, test_df):
    """
    Ensure test data has exactly the same features as training data
    in the same order.
    """
    # 1. Add missing columns with 0
    missing_cols = set(train_columns) - set(test_df.columns)
    for col in missing_cols:
        test_df[col] = 0
        print(f"⚠️ Added missing column '{col}' filled with 0")

    # 2. Remove extra columns
    extra_cols = set(test_df.columns) - set(train_columns)
    if extra_cols:
        print(f"⚠️ Removing extra columns: {extra_cols}")
        test_df = test_df[train_columns]  # Select only the necessary columns

    # 3. Ensure correct order
    test_df = test_df[train_columns]  # Reorder columns to match training data

    return test_df

# Usage:
# Get the feature names the model was trained on
train_columns = clf.get_booster().feature_names

# Align test data features
# Use test_df instead of the undefined test_df_processed
test_df_aligned = align_test_features(train_columns, test_df)

# Now predict
pred = clf.predict(test_df_aligned)
```
### 25.  Mempersiapkan Submission
Membuat DataFrame submissions berisi kolom id dan hasil prediksi target.
``` py
submissions = pd.DataFrame({'id': test['id'], 'target': pred})
submissions.head()
```
### 26. Menyimpan Submission ke CSV
Menyimpan DataFrame submissions ke file `submissiongajelas9.csv`.
``` py
import pandas as pd

# Create the 'submissions' DataFrame
submissions = pd.DataFrame({'id': test['id'], 'target': pred})

# Display the first few rows of 'submissions' to verify it's created correctly
submissions.head()

# Save the DataFrame to CSV
submissions.to_csv('submissiongajelas9.csv', index=False)
```

