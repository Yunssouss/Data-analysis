# استيراد المكتبات اللازمة
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# إنشاء جدول البيانات
data = {
    'Month': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
    'Sales': [150, 200, 180, 250, 220, 210, 190, 230, 210, 240, 280, 300],
    'Product Category': ['A', 'A', 'B', 'A', 'B', 'B', 'C', 'C', 'A', 'B', 'C', 'A'],
    'Profit': [50, 80, 60, 100, 90, 85, 75, 95, 80, 110, 130, 150]
}

df = pd.DataFrame(data)

# تحويل الشهور لأرقام
month_mapping = {
    "January": 1, "February": 2, "March": 3, "April": 4, 
    "May": 5, "June": 6, "July": 7, "August": 8, 
    "September": 9, "October": 10, "November": 11, "December": 12
}
df['Month'] = df['Month'].map(month_mapping)

# تحويل القيم النصية في عمود 'Product Category' إلى قيم رقمية
label_encoder = LabelEncoder()
df['Product Category'] = label_encoder.fit_transform(df['Product Category'])

# تحديد المميزات (X) والهدف (y)
X = df.drop(['Profit'], axis=1)  # 'Profit' هو العمود الهدف اللي بغينا نتنبؤو بيه
y = df['Profit']

# تقسيم البيانات إلى مجموعات التدريب والاختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# إنشاء نموذج Gradient Boosting Regressor
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.01, max_depth=2,min_samples_split=5 ,random_state=42)

# تدريب النموذج باستخدام مجموعة التدريب
model.fit(X_train, y_train)

# توقع النتائج باستخدام مجموعة الاختبار
y_pred = model.predict(X_test)

# تقييم النموذج عبر حساب Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
