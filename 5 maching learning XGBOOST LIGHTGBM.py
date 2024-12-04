import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



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

from sklearn.model_selection import train_test_split

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'reg:squarederror',  # خاصنا مقياس مربع الخطأ
    'learning_rate': 0.1,  # معدل التعلم
    'max_depth': 5,  # العمق ديال الأشجار
    'alpha': 10,  # L1 regularization للتقليل من overfitting
    'n_estimators': 100  # عدد الأشجار
}

model = xgb.train(params, dtrain, num_boost_round=100)

predictions = model.predict(dtest)

mse = mean_squared_error(y_test, predictions)
print(f'XGBoost MSE: {mse}')

future_data = pd.DataFrame({
    'month': [10, 11, 12],  # شهور المستقبل
    'profit': [120, 130, 140],  # أرباح متوقعة
    'product_category': [0, 1, 0]  # الفئة
})

# تحويل المستقبل DMatrix
dfuture = xgb.DMatrix(future_data)

# توقع المبيعات باستعمال XGBoost
future_predictions = model.predict(dfuture)
print(f'توقع المبيعات باستخدام XGBoost: {future_predictions}')

from sklearn.model_selection import GridSearchCV

param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'alpha': [1, 10, 100]
}

grid_search = GridSearchCV(estimator=xgb.XGBRegressor(n_estimators=100), param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

print(f'Best parameters: {grid_search.best_params_}')

