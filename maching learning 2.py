import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


# توليد التواريخ بشكل شهري من بداية 2023 حتى نهاية أكتوبر 2024
dates = pd.date_range(start='2023-01-01', periods=22, freq='M')

# بيانات المبيعات الشهرية
monthly_sales = [5000, 5200, 5300, 5400, 6000, 6200, 5800, 6100, 6500, 6300, 6400, 6700,  # مبيعات 2023
                 6800, 6900, 7000, 7100, 7300, 7400, 7600, 7800, 7700, 8000]  # مبيعات 2024 حتى الشهر 10

# إنشاء DataFrame بالبيانات
data = pd.DataFrame({
    'date': dates,
    'product_id': [1] * len(dates),  # منتوج واحد كمثال
    'quantity_sold': monthly_sales,
    'sales_amount': [q * np.random.uniform(1.1, 1.3) for q in monthly_sales]  # تضخيم طفيف للمبيعات
})

# 2. نظرة أولية على البيانات
print("ملخص البيانات:\n", data.describe())

sns.lineplot(data=data, x='date', y='sales_amount')
plt.title('تطور المبيعات الشهرية')
plt.xlabel('الشهر')
plt.ylabel('قيمة المبيعات')
plt.show()

# 3. إعداد البيانات للنموذج
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year
X = data[['month', 'year', 'quantity_sold']]
y = data['sales_amount']

# 4. بناء النموذج ديال XGBoost
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
xgb_model.fit(X, y)

# 5. التنبؤ بالمبيعات ديال الشهر 11
future_data = pd.DataFrame({
    'month': [11],
    'year': [2024],
    'quantity_sold': [8200]  # توقع مبيعات عادية للشهر 11
})
y_pred = xgb_model.predict(future_data)

print("توقع المبيعات للشهر 11:", y_pred[0])

# 6. حساب مؤشرات الدقة MSE, RMSE, MAE
y_actual = data[data['month'] == 10]['sales_amount']  # مبيعات حقيقية للشهر 10
y_predict = xgb_model.predict(X[data['month'] == 10])

mse = mean_squared_error(y_actual, y_predict)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_actual, y_predict)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)

# 7. عرض النتائج
plt.plot(dates, data['sales_amount'], label='Actual Sales')
plt.plot(dates, xgb_model.predict(X), label='Predicted Sales', linestyle='--')
plt.legend()
plt.title('Actual vs Predicted Sales')
plt.xlabel('Date')
plt.ylabel('Sales Amount')
plt.show()
