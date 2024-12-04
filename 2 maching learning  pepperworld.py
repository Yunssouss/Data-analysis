import pandas as pd
import numpy as np

# إنشاء البيانات الفلاحية لـ 12 شهر
data = {
    'month': ['January', 'February', 'March', 'April', 'May', 'June', 
              'July', 'August', 'September', 'October', 'November', 'December'],
    'sales': [150, 200, 180, 250, 220, 210, 230, 240, 260, 270, np.nan, np.nan],
    'profit': [50, 80, 60, 100, 90, 110, 95, 105, 120, 130, 140, 150],
    'product_category': ['Green Pepper', 'Red Pepper', 'Sweet Pepper', 
                         'Green Pepper', 'Red Pepper', 'Sweet Pepper', 
                         'Green Pepper', 'Red Pepper', 'Sweet Pepper',
                         'Green Pepper', 'Red Pepper', 'Sweet Pepper']
}

df = pd.DataFrame(data)

# تعويض القيم المفقودة فـ sales بالمتوسط
df['sales'].fillna(df['sales'].mean(), inplace=True)

print(df)


df = pd.DataFrame(data)

# تعويض القيم المفقودة فـ sales بالمتوسط
df['sales'].fillna(df['sales'].mean(), inplace=True)

# تحويل الشهور إلى أرقام
df['month'] = pd.to_datetime(df['month'], format='%B').dt.month

# تحويل فئة المنتج إلى أرقام
df['product_category'] = df['product_category'].apply(lambda x: 0 if x == 'Green Pepper' else (1 if x == 'Red Pepper' else 2))

print(df)

# المتغيرات المستقلة (features)
X = df[['month', 'profit', 'product_category']]

# المتغير التابع (المبيعات)
y = df['sales']

from sklearn.model_selection import train_test_split

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.neighbors import KNeighborsRegressor

# إنشاء وتدريب موديل KNN
knn_model = KNeighborsRegressor(n_neighbors=3)
knn_model.fit(X_train, y_train)

# التوقع
knn_predictions = knn_model.predict(X_test)

from sklearn.metrics import mean_squared_error
knn_mse = mean_squared_error(y_test, knn_predictions)
print(f'KNN MSE: {knn_mse}')

from sklearn.ensemble import RandomForestRegressor

# تدريب الموديل
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# توقع المبيعات المستقبلية باستعمال بيانات الاختبار
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
    'month': [10, 11, 12],  # شهور المستقبل
    'profit': [150, 160, 170],  # أرباح متوقعة
    'product_category': [0, 1, 2]  # الفئة (Green Pepper, Red Pepper, Sweet Pepper)
})

# توقع المبيعات المستقبلية باستعمال Random Forest
future_predictions_rf = rf_model.predict(future_data)
print(f'توقع المبيعات باستخدام Random Forest: {future_predictions_rf}')

# توقع المبيعات المستقبلية باستعمال Gradient Boosting
future_predictions_gb = gb_model.predict(future_data)
print(f'توقع المبيعات باستخدام Gradient Boosting: {future_predictions_gb}')

