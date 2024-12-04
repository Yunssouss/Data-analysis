import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


# 1. إعداد البيانات
dates = pd.date_range(start='2022-01-01', end='2024-10-31', freq='M')
products = ['تفاح', 'برتقال', 'موز', 'فراولة']

# إنشاء بيانات وهمية
data = []
np.random.seed(0)
for product in products:
    sales = np.random.normal(loc=5000, scale=800, size=len(dates))
    quantity = np.random.normal(loc=100, scale=20, size=len(dates))
    for date, q, s in zip(dates, quantity, sales):
        data.append([date, product, abs(int(q)), abs(float(s))])

# تحويل البيانات ل DataFrame
df = pd.DataFrame(data, columns=['date', 'product', 'quantity_sold', 'sales_amount'])

print("ملخص البيانات:\n", df.describe())

# استعراض المبيعات الشهرية لكل منتوج
plt.figure(figsize=(14, 7))
sns.lineplot(data=df, x='date', y='sales_amount', hue='product')
plt.title('مبيعات المنتجات شهريًا')
plt.xlabel('التاريخ')
plt.ylabel('قيمة المبيعات')
plt.show()

# اختيار منتوج للتنبؤ بمبيعاته
product_data = df[df['product'] == 'تفاح'].copy()

# تحويل البيانات للتنبؤات
product_data['month'] = product_data['date'].dt.month
product_data['year'] = product_data['date'].dt.year

# تدريب النموذج
X = product_data[['month', 'year']]
y = product_data['sales_amount']

# تقسيم البيانات
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# تدريب نموذج XGBoost
model_xgb = XGBRegressor()
model_xgb.fit(X_train, y_train)

# التوقعات على بيانات الاختبار
y_pred_xgb = model_xgb.predict(X_test)

# عرض نتائج التوقعات لبعض العينات في بيانات الاختبار
print("التوقعات ديال المبيعات باستعمال XGBoost:")
print("التوقعات:", y_pred_xgb[:10])  # عرض أول 10 توقعات
print("القيم الفعلية:", y_test[:10].values)  # عرض أول 10 قيم فعلية للمقارنة

# حساب مقاييس التقييم
mse = mean_squared_error(y_test, y_pred_xgb)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_xgb)

# عرض مقاييس الخطأ
print("\nنتائج التقييم لنموذج XGBoost:")
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)


from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

# تحديد قيم المعلمات لي بغا نختبرو
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# إنشاء كائن GridSearchCV
grid_search = GridSearchCV(
    estimator=XGBRegressor(),
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,  # عدد تقسيمات Cross-validation
    verbose=1,
    n_jobs=-1  # استعمال جميع أنوية المعالج
)

# تشغيل Grid Search
grid_search.fit(X_train, y_train)

# عرض أفضل معلمات
print("أفضل معلمات:", grid_search.best_params_)
print("MSE لأفضل نموذج:", -grid_search.best_score_)


from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from scipy.stats import uniform

# تحديد توزيع المعلمات لي بغا نختبرو
param_dist = {
    'n_estimators': [int(x) for x in range(100, 301, 50)],
    'learning_rate': uniform(0.01, 0.3),
    'max_depth': [3, 5, 7],
    'subsample': uniform(0.7, 0.3),
    'colsample_bytree': uniform(0.7, 0.3)
}

# إنشاء كائن RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=XGBRegressor(),
    param_distributions=param_dist,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=1,
    n_iter=50,  # عدد المحاولات
    n_jobs=-1
)

# تشغيل Random Search
random_search.fit(X_train, y_train)

# عرض أفضل معلمات
print("أفضل معلمات:", random_search.best_params_)
print("MSE لأفضل نموذج:", -random_search.best_score_)
