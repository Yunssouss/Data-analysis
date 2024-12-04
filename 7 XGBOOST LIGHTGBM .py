import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


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



from xgboost import XGBRegressor

# إنشاء وتدريب موديل XGBoost
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# التوقع
xgb_predictions = xgb_model.predict(X_test)

# حساب MSE ديال XGBoost
xgb_mse = mean_squared_error(y_test, xgb_predictions)
print(f'XGBoost MSE: {xgb_mse}')


from lightgbm import LGBMRegressor

# إنشاء وتدريب موديل LightGBM
lgb_model = LGBMRegressor(n_estimators=100, random_state=42)
lgb_model.fit(X_train, y_train)

# التوقع
lgb_predictions = lgb_model.predict(X_test)

# حساب MSE ديال LightGBM
lgb_mse = mean_squared_error(y_test, lgb_predictions)
print(f'LightGBM MSE: {lgb_mse}')

# مثال لتحضير بيانات المستقبل
future_data = pd.DataFrame({
    'month': [13, 14, 15],  # هاد الأرقام تمثل الشهور الجاية
    'product_category': [0, 1, 2]  # قيم الفئات نفس اللي درتي في التدريب
})

# توقع باستخدام نموذج XGBoost
future_predictions_xgb = xgb_model.predict(future_data)
print(f'توقع المبيعات باستخدام XGBoost: {future_predictions_xgb}')

# توقع المبيعات المستقبلية باستعمال XGBoost
future_predictions_xgb = xgb_model.predict(future_data)
print(f'توقع المبيعات باستخدام XGBoost: {future_predictions_xgb}')

# توقع المبيعات المستقبلية باستعمال LightGBM
future_predictions_lgb = lgb_model.predict(future_data)
print(f'توقع المبيعات باستخدام LightGBM: {future_predictions_lgb}')

from xgboost import XGBClassifier

# تحويل المبيعات إلى فئات (منخفضة، متوسطة، عالية)
df['sales_category'] = pd.cut(df['sales'], bins=3, labels=[0, 1, 2])

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, df['sales_category'], test_size=0.2, random_state=42)

# إنشاء وتدريب موديل XGBoost للتصنيف
xgb_class_model = XGBClassifier(n_estimators=100, random_state=42)
xgb_class_model.fit(X_train, y_train)

# توقع الفئات
xgb_class_predictions = xgb_class_model.predict(X_test)

# حساب دقة الموديل
from sklearn.metrics import accuracy_score
xgb_class_accuracy = accuracy_score(y_test, xgb_class_predictions)
print(f'XGBoost Classification Accuracy: {xgb_class_accuracy}')

from lightgbm import LGBMClassifier

# إنشاء وتدريب موديل LightGBM للتصنيف
lgb_class_model = LGBMClassifier(n_estimators=100, random_state=42)
lgb_class_model.fit(X_train, y_train)

# توقع الفئات
lgb_class_predictions = lgb_class_model.predict(X_test)

# حساب دقة الموديل
lgb_class_accuracy = accuracy_score(y_test, lgb_class_predictions)
print(f'LightGBM Classification Accuracy: {lgb_class_accuracy}')



