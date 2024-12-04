import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsRegressor

# إنشاء البيانات الفلاحية لـ 9 أشهر
data = {
    'month': ['January', 'February', 'March', 'April', 'May', 'June', 
              'July', 'August', 'September'],
    'sales': [150, 200, 180, 250, 220, 210, 230, 240, 260],
    'profit': [50, 80, 60, 100, 90, 110, 95, 105, 120],
    'product_category': ['Green Pepper', 'Red Pepper', 'Sweet Pepper', 
                         'Green Pepper', 'Red Pepper', 'Sweet Pepper', 
                         'Green Pepper', 'Red Pepper', 'Sweet Pepper']
}

df = pd.DataFrame(data)

print(df)

# تحويل الشهور لأرقام
df['month'] = pd.to_datetime(df['month'], format='%B').dt.month

# تحويل فئة المنتج لأرقام
df['product_category'] = df['product_category'].apply(lambda x: 0 if x == 'Green Pepper' else (1 if x == 'Red Pepper' else 2))

# المتغيرات المستقلة (features)
X = df[['month', 'profit', 'product_category']]

# المتغير التابع (المبيعات)
y = df['sales']

# تقسيم البيانات للتدريب والاختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب موديل Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=300,max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# توقع المبيعات باستعمال Random Forest Regressor
rf_predictions = rf_model.predict(X_test)

# حساب MSE
rf_mse = mean_squared_error(y_test, rf_predictions)
print(f'Random Forest Regressor MSE: {rf_mse}')

# تدريب موديل Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

# توقع المبيعات باستعمال Gradient Boosting Regressor
gb_predictions = gb_model.predict(X_test)

# حساب MSE
gb_mse = mean_squared_error(y_test, gb_predictions)
print(f'Gradient Boosting Regressor MSE: {gb_mse}')

# البيانات المستقبلية للتوقع
future_data = pd.DataFrame({
    'month': [10, 11, 12],  # شهور المستقبل
    'profit': [150, 160, 170],  # أرباح متوقعة
    'product_category': [0, 1, 2]  # الفئة (Green Pepper, Red Pepper, Sweet Pepper)
})

# توقع المبيعات المستقبلية باستعمال Random Forest
future_predictions_rf = rf_model.predict(future_data)
print(f'توقع المبيعات باستخدام Random Forest Regressor: {future_predictions_rf}')

# توقع المبيعات المستقبلية باستعمال Gradient Boosting
future_predictions_gb = gb_model.predict(future_data)
print(f'توقع المبيعات باستخدام Gradient Boosting Regressor: {future_predictions_gb}')

# الآن نديرو التصنيف باستعمال Random Forest Classifier و Gradient Boosting Classifier
# إنشاء فئة للمبيعات (نقولوا مثلا المبيعات العالية هي أكثر من 220)
df['high_sales'] = df['sales'].apply(lambda x: 1 if x > 220 else 0)

# المتغيرات المستقلة للتصنيف
X_classification = df[['month', 'profit', 'product_category']]
y_classification = df['high_sales']

# تقسيم البيانات
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)

# تدريب موديل Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_c, y_train_c)

# توقع التصنيف
rf_class_predictions = rf_classifier.predict(X_test_c)


# تدريب موديل Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_classifier.fit(X_train_c, y_train_c)

# توقع التصنيف
gb_class_predictions = gb_classifier.predict(X_test_c)


from sklearn.metrics import accuracy_score

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
