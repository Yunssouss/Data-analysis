import pandas as pd
import numpy as np

# إنشاء بيانات كبيرة بحال الواقع
np.random.seed(42)
data = pd.DataFrame({
    'date': pd.date_range(start='2021-01-01', periods=100000, freq='H'),  # تواريخ ديال عامين
    'product_id': np.random.randint(1, 100, 100000),   # معرفات ديال المنتوجات
    'quantity_sold': np.random.randint(1, 10, 100000), # الكمية المباعة
    'sales_amount': np.random.uniform(5, 100, 100000)  # ثمن البيع فدرهم
})

# نتاكدو من القيم الفارغة ونحيدوهم
data = data.dropna()

# نتاكدو من التكرار ونحيدوه
data = data.drop_duplicates()

# نشوفو الإحصائيات الرئيسية
print(data.describe())

# توزيع عدد مرات بيع المنتوجات
print(data['product_id'].value_counts().head(10))

# حساب مؤشرات أساسية بحال المتوسط والوسيط
mean_sales = data['sales_amount'].mean()
median_sales = data['sales_amount'].median()
std_sales = data['sales_amount'].std()

print(f"متوسط المبيعات: {mean_sales}, الوسيط: {median_sales}, الانحراف المعياري: {std_sales}")

# استخراج الشهر من التاريخ
data['month'] = data['date'].dt.month
monthly_sales = data.groupby(['month', 'product_id'])['sales_amount'].sum().reset_index()
print(monthly_sales.head())


# حساب عدد مرات بيع كل منتوج شهريا
monthly_sales['sales_count'] = data.groupby(['month', 'product_id'])['quantity_sold'].sum().values
print(monthly_sales.head())

# تحديد المتغيرات المستقلة (X) والتابعة (y)
X = monthly_sales[['month', 'product_id', 'sales_count']]
y = monthly_sales['sales_amount']

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب النموذج
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# التوقع على البيانات ديال الاختبار
y_pred = model.predict(X_test)

# حساب الخطأ
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")






