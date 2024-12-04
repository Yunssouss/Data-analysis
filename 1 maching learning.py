import pandas as pd
import numpy as np

# إنشاء البيانات اللي غنخدمو عليها
data = {
    'month': ['January', 'February', 'March', 'April', 'May', 'June'],
    'sales': [150, 200, 180, 250, 220, np.nan],
    'profit': [50, 80, 60, 100, 90, 110],
    'product_category': ['A', 'A', 'B', 'A', 'B', 'A']
}

df = pd.DataFrame(data)

print(df)

# تعويض القيم المفقودة فـ sales بالمتوسط
df['sales'].fillna(df['sales'].mean(), inplace=True)

print(df)
 
 # تحويل الشهور إلى أرقام
df['month'] = pd.to_datetime(df['month'], format='%B').dt.month

# تحويل فئة المنتج لأرقام (A = 0, B = 1)
df['product_category'] = df['product_category'].apply(lambda x: 0 if x == 'A' else 1)

print(df)

# المتغيرات المستقلة (features)
X = df[['month', 'profit', 'product_category']]

# المتغير التابع (المبيعات)
y = df['sales']

from sklearn.model_selection import train_test_split

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.neighbors import KNeighborsRegressor

# إنشاء الموديل ديال KNN
knn_model = KNeighborsRegressor(n_neighbors=3)

# تدريب الموديل
knn_model.fit(X_train, y_train)

# توقع المبيعات باستعمال بيانات الاختبار
knn_predictions = knn_model.predict(X_test)

print("توقعات KNN:", knn_predictions)
 
from sklearn.metrics import mean_squared_error

# حساب MSE ديال KNN
knn_mse = mean_squared_error(y_test, knn_predictions)
print(f'KNN MSE: {knn_mse}')

from sklearn.linear_model import LinearRegression

# إنشاء الموديل
linear_model = LinearRegression()

# تدريب الموديل
linear_model.fit(X_train, y_train)

# توقع المبيعات
linear_predictions = linear_model.predict(X_test)

print("توقعات Linear Regression:", linear_predictions)

from sklearn.metrics import mean_squared_error

# حساب MSE ديال Linear Regression
linear_mse = mean_squared_error(y_test, linear_predictions)
print(f'Linear Regression MSE: {linear_mse}')

from sklearn.tree import DecisionTreeRegressor

# إنشاء الموديل ديال Decision Trees
tree_model = DecisionTreeRegressor(random_state=42)

# تدريب الموديل
tree_model.fit(X_train, y_train)

# توقع المبيعات
tree_predictions = tree_model.predict(X_test)

print("توقعات Decision Trees:", tree_predictions)

# حساب MSE ديال Decision Trees
tree_mse = mean_squared_error(y_test, tree_predictions)
print(f'Decision Tree MSE: {tree_mse}')

# تحويل sales لفئات (تصنيف)
df['sales_category'] = df['sales'].apply(lambda x: 1 if x > 200 else 0)

# المتغيرات الجديدة
X = df[['month', 'profit', 'product_category']]
y = df['sales_category']

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

# إنشاء الموديل ديال Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# تدريب الموديل
rf_classifier.fit(X_train, y_train)

# توقع الفئات
rf_predictions = rf_classifier.predict(X_test)

print("توقعات Random Forest Classifier:", rf_predictions)

from sklearn.metrics import accuracy_score

# حساب الدقة (accuracy) ديال Random Forest Classifier
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f'Random Forest Accuracy: {rf_accuracy}')

from sklearn.ensemble import GradientBoostingClassifier

# إنشاء الموديل ديال Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# تدريب الموديل
gb_classifier.fit(X_train, y_train)

# توقع الفئات
gb_predictions = gb_classifier.predict(X_test)

print("توقعات Gradient Boosting Classifier:", gb_predictions)

# حساب الدقة ديال Gradient Boosting Classifier
gb_accuracy = accuracy_score(y_test, gb_predictions)
print(f'Gradient Boosting Accuracy: {gb_accuracy}')

import pandas as pd
import sqlite3

# اتصال مع قاعدة البيانات ديال sqlite
conn = sqlite3.connect('monthly_sales.db')

# استخراج البيانات من قاعدة البيانات
query = "SELECT month, sales, profit, product_category FROM sales"
df = pd.read_sql_query(query, conn)

# إغلاق الاتصال
conn.close()

# عرض البيانات
print(df.head())

# تحويل الشهور لأرقام
df['month'] = pd.to_datetime(df['month'], format='%B').dt.month

# المتغيرات المستقلة (features)
X = df[['month', 'profit', 'product_category']]

# تحويل فئة المنتج لأرقام
X['product_category'] = X['product_category'].apply(lambda x: 0 if x == 'A' else 1)

# المتغير التابع (المبيعات)
y = df['sales']

from sklearn.model_selection import train_test_split

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor

# تدريب الموديل
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# توقع المبيعات المستقبلية باستعمال البيانات ديال الاختبار
rf_predictions = rf_model.predict(X_test)

from sklearn.ensemble import GradientBoostingRegressor

# تدريب الموديل
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

# توقع المبيعات المستقبلية
gb_predictions = gb_model.predict(X_test)

from sklearn.metrics import mean_squared_error

# تقييم الموديل ديال Random Forest
rf_mse = mean_squared_error(y_test, rf_predictions)
print(f'Random Forest MSE: {rf_mse}')

# تقييم الموديل ديال Gradient Boosting
gb_mse = mean_squared_error(y_test, gb_predictions)
print(f'Gradient Boosting MSE: {gb_mse}')

# الشهور اللي غنبغيو نتوقعو ليهم
future_data = pd.DataFrame({
    'month': [6, 7, 8  ],  # شهور المستقبل
    'profit': [120, 130, 140],  # أرباح متوقعة
    'product_category': [0, 1, 0]  # الفئة
})

# توقع المبيعات المستقبلية باستعمال Random Forest
future_predictions_rf = rf_model.predict(future_data)
print(f'توقع المبيعات باستخدام Random Forest: {future_predictions_rf}')

# توقع المبيعات المستقبلية باستعمال Gradient Boosting
future_predictions_gb = gb_model.predict(future_data)
print(f'توقع المبيعات باستخدام Gradient Boosting: {future_predictions_gb}') 


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd


# نجيب البيانات ديالنا
df = pd.DataFrame({
    'month': [1, 2, 3, 4, 5, 6],
    'sales': [150, 200, 180, 250, 220, 200],
    'profit': [50, 80, 60, 100, 90, 110],
    'product_category': [0, 0, 1, 0, 1, 0]
})

# توحيد البيانات
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['sales', 'profit']])

# تطبيق PCA
pca = PCA(n_components=1)
df_pca = pca.fit_transform(df_scaled)

print("البيانات من بعد PCA: ", df_pca)


from sklearn.model_selection import train_test_split, GridSearchCV,LeaveOneOut
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# تحضير البيانات
X = df[['sales', 'profit', 'product_category']]
y = [0, 0, 1, 0, 1, 0]  # التصنيفات

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# تطبيق SVM
svm = SVC()
svm.fit(X_train, y_train)

# التنبؤ بالنتائج
y_pred = svm.predict(X_test)
print("دقة SVM: ", accuracy_score(y_test, y_pred))


# Grid Search على SVM
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
loo=LeaveOneOut()
# Grid Search مع LOOCV
grid = GridSearchCV(SVC(), param_grid, cv=loo, refit=True, verbose=2)

grid = GridSearchCV(SVC(),param_grid,cv=2 ,refit=True, verbose=2)
grid.fit(X_train, y_train)

print("أفضل إعدادات Grid Search: ", grid.best_params_)

from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

# تعريف المعايير اللي غادين نجربوها فـRandomizedSearch
param_dist = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

# تقليل عدد الـsplits فـcross validation لـ2
random = RandomizedSearchCV(SVC(), param_distributions=param_dist, n_iter=5, cv=2, refit=True, verbose=2)

# تدريب الموديل
random.fit(X_train, y_train)

# عرض أفضل الإعدادات
print("أفضل إعدادات Randomized Search: ", random.best_params_)

from sklearn.model_selection import LeaveOneOut, RandomizedSearchCV
from sklearn.svm import SVC

# تعريف المعايير
param_dist = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

# استخدام LeaveOneOut
loo = LeaveOneOut()

# RandomizedSearchCV مع LeaveOneOut
random = RandomizedSearchCV(SVC(), param_distributions=param_dist, n_iter=5, cv=loo, refit=True, verbose=2)

# تدريب الموديل
random.fit(X_train, y_train)

# عرض أفضل الإعدادات
print("أفضل إعدادات Randomized Search: ", random.best_params_)


import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# البيانات ديالك
df = pd.DataFrame({
    'month': [1, 2, 3, 4, 5, 6],
    'sales': [150, 200, 180, 250, 220, 200],
    'profit': [50, 80, 60, 100, 90, 110],
    'product_category': [0, 0, 1, 0, 1, 0]
})

# نضيف عمود ديال نسبة الربح إلى المبيعات
df['profit_to_sales'] = df['profit'] / df['sales']

# المتغيرات المستقلة والتابعة
X = df[['month', 'sales', 'profit', 'profit_to_sales']]
y = df['product_category']

# نقسم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# نطبق SMOTE باش نزيدو البيانات
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# تدريب SVM بعد زيادة البيانات
svc = SVC()
svc.fit(X_train_smote, y_train_smote)

# التوقع
y_pred = svc.predict(X_test)

# حساب الدقة
accuracy = accuracy_score(y_test, y_pred)
print("دقة النموذج بعد SMOTE: ", accuracy)





























