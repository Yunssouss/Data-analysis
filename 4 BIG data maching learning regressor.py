import pandas as pd
import numpy as np

# إنشاء بيانات فلاحية جديدة للفلفلة من 2001 إلى 2024
years = np.arange(2001, 2025)
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
categories = ['Green Pepper', 'Red Pepper', 'Sweet Pepper']

# إنشاء بيانات عشوائية للمبيعات والأرباح
data = []

for year in years:
    for month in months:
        for category in categories:
            sales = np.random.randint(100, 300)  # مبيعات عشوائية بين 100 و 300
            profit = np.random.randint(50, 150)  # أرباح عشوائية بين 50 و 150
            data.append([f"{month} {year}", sales, profit, category])

# تحويل البيانات إلى DataFrame
df = pd.DataFrame(data, columns=['month', 'sales', 'profit', 'product_category'])



print(df)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# إنشاء البيانات الفلاحية لمنتوج الفلفلة من 2001 حتى 2024
years = list(range(2001, 2025))
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
product_categories = ['Green Pepper', 'Red Pepper', 'Sweet Pepper']

data = []
for year in years:
    for month in months:
        for category in product_categories:
            sales = np.random.randint(100, 300)  # مبيعات عشوائية بين 100 و 300
            profit = np.random.randint(50, 150)  # أرباح عشوائية بين 50 و 150
            data.append([f"{month} {year}", sales, profit, category])

df = pd.DataFrame(data, columns=['month', 'sales', 'profit', 'product_category'])

# تحويل الشهور إلى أرقام
df['month'] = pd.to_datetime(df['month'], format='%B %Y').dt.month

# تحويل فئة المنتج إلى أرقام
df['product_category'] = df['product_category'].apply(lambda x: 0 if x == 'Green Pepper' else (1 if x == 'Red Pepper' else 2))

# المتغيرات المستقلة (features)
X = df[['month', 'profit', 'product_category']]

# المتغير التابع (المبيعات)
y = df['sales']

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# توقع المبيعات باستخدام بيانات الاختبار
rf_predictions = rf_model.predict(X_test)

# حساب MSE
rf_mse = mean_squared_error(y_test, rf_predictions)
print(f'Random Forest Regressor MSE: {rf_mse}')

# تدريب Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

# توقع المبيعات باستخدام بيانات الاختبار
gb_predictions = gb_model.predict(X_test)

# حساب MSE
gb_mse = mean_squared_error(y_test, gb_predictions)
print(f'Gradient Boosting Regressor MSE: {gb_mse}')

# توقع المبيعات للأشهر القادمة (أكتوبر، نونبر، دجنبر 2024)
future_data = pd.DataFrame({
    'month': [10, 11, 12],  # شهور المستقبل
    'profit': [150, 160, 170],  # أرباح متوقعة
    'product_category': [0, 1, 2]  # الفئة (Green Pepper, Red Pepper, Sweet Pepper)
})

# توقع المبيعات باستخدام Random Forest Regressor
future_predictions_rf = rf_model.predict(future_data)
print(f'توقع المبيعات باستخدام Random Forest Regressor: {future_predictions_rf}')

# توقع المبيعات باستخدام Gradient Boosting Regressor
future_predictions_gb = gb_model.predict(future_data)
print(f'توقع المبيعات باستخدام Gradient Boosting Regressor: {future_predictions_gb}')













