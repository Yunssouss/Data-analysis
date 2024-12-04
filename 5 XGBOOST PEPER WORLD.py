import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# البيانات الفلاحية
data = {
    'month': ['January', 'February', 'March', 'April', 'May', 'June', 
              'July', 'August', 'September'],
    'sales': [150, 200, 180, 250, 220, 210, 230, 240, 260],
    'profit': [50, 80, 60, 100, 90, 110, 95, 105, 120],
    'product_category': ['Green Pepper', 'Red Pepper', 'Sweet Pepper', 
                         'Green Pepper', 'Red Pepper', 'Sweet Pepper', 
                         'Green Pepper', 'Red Pepper', 'Sweet Pepper']
}

# تحويل الداتا إلى DataFrame
df = pd.DataFrame(data)

# استعمال one-hot encoding على فئة المنتوجات
df = pd.get_dummies(df, columns=['product_category'], drop_first=True)

# تحويل عمود "month" باستخدام LabelEncoder
label_encoder = LabelEncoder()
df['month'] = label_encoder.fit_transform(df['month'])

# تطبيق MinMaxScaler على sales و profit
scaler = MinMaxScaler()
df[['sales', 'profit']] = scaler.fit_transform(df[['sales', 'profit']])

# تقسيم البيانات
X = df.drop('sales', axis=1)  # حذف عمود 'sales' لأنه هو الهدف
y = df['sales']

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# نموذج XGBoost
xgb_model = XGBRegressor(alpha=1, learning_rate=0.01, max_depth=3)

# استعمال cross-validation
scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

# التدريب على البيانات
xgb_model.fit(X_train, y_train)

# التوقعات على بيانات الاختبار
y_pred = xgb_model.predict(X_test)

# حساب الـ MSE
mse = mean_squared_error(y_test, y_pred)

# عرض النتائج
print("MSE:", mse)
print("Cross-Validation Scores:", scores)

