import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor, DMatrix
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. تحضير البيانات
dates = pd.date_range(start='2023-01-01', end='2024-10-31', freq='M')
data = pd.DataFrame({
    'date': dates,
    'product_id': np.random.randint(1, 10, len(dates)),
    'quantity_sold': np.random.randint(100, 1000, len(dates)),
    'sales_amount': np.random.uniform(1000, 10000, len(dates))
})

# 2. نظرة أولية على البيانات
print("ملخص البيانات:\n", data.describe())
sns.lineplot(data=data, x='date', y='sales_amount')
plt.title('مبيعات الشهرية ديال العامين')
plt.show()

# 3. إعداد البيانات
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year
X = data[['month', 'year', 'product_id', 'quantity_sold']]
y = data['sales_amount']

# 4. بناء النموذج ديال XGBoost
xgb_model = XGBRegressor(objective='reg:squarederror')
xgb_model.fit(X, y)

# 5. التنبؤ بالمبيعات ديال الشهر 11
future_data = pd.DataFrame({
    'month': [11],
    'year': [2024],
    'product_id': [5],  # مثلاً منتوج معين
    'quantity_sold': [np.random.randint(100, 1000)]
})
y_pred = xgb_model.predict(future_data)

print("توقع المبيعات للشهر 11:", y_pred[0])

# 6. حساب مؤشرات الدقة MSE, RMSE, MAE
y_actual = data[data['date'].dt.month == 10]['sales_amount']  # داتا حقيقية للشهر 10
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
