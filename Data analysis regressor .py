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
    'month': [6, 7, 8],  # شهور المستقبل
    'profit': [120, 130, 140],  # أرباح متوقعة
    'product_category': [0, 1, 0]  # الفئة
})

# توقع المبيعات المستقبلية باستعمال Random Forest
future_predictions_rf = rf_model.predict(future_data)
print(f'توقع المبيعات باستخدام Random Forest: {future_predictions_rf}')

# توقع المبيعات المستقبلية باستعمال Gradient Boosting
future_predictions_gb = gb_model.predict(future_data)
print(f'توقع المبيعات باستخدام Gradient Boosting: {future_predictions_gb}') 






