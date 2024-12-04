import pandas as pd

# إنشاء البيانات
data = {
    'Month': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
    'Sales': [150, 200, 180, 250, 220, 210, 190, 230, 210, 240, 280, 300],
    'Product Category': ['A', 'A', 'B', 'A', 'B', 'B', 'C', 'C', 'A', 'B', 'C', 'A'],
    'Profit': [50, 80, 60, 100, 90, 85, 75, 95, 80, 110, 130, 150]
}

# إنشاء DataFrame
df = pd.DataFrame(data)

# تحويل الشهور لأرقام
month_mapping = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5,
    'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10,
    'November': 11, 'December': 12
}

df['Month'] = df['Month'].map(month_mapping)

# تحويل فئة المنتجات لـ One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['Product Category'])

# عرض الـ DataFrame بعد التحويل
print(df_encoded.head())

# الآن الميزات (X) هي كل شيء ما عدا المبيعات
X = df_encoded.drop('Sales', axis=1)

# المتغير المستهدف (y) هو المبيعات
y = df_encoded['Sales']

# عرض الأعمدة الجديدة والتأكد أن كل شيء على ما يرام
print(X.columns)  


import seaborn as sns
import matplotlib.pyplot as plt

# حساب الارتباط بين المبيعات والأرباح
# إزالة الأعمدة غير الرقمية قبل حساب الارتباط
correlation = df.drop(['Month', 'Product Category'], axis=1).corr()


# عرض خريطة الارتباط
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# تحويل فئة المنتجات ل One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['Product Category'])

# استعراض البيانات بعد التحويل
print(df_encoded.head())

# الميزات (X) هي الأرباح وفئة المنتجات
X = df_encoded.drop('Sales', axis=1)

# المتغير المستهدف (y) هو المبيعات
y = df_encoded['Sales']

# استعراض الميزات والمتغير المستهدف
print(X.head())
print(y.head())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# تحويل True/False لأرقام 0 و 1
df['Product Category_A'] = df['Product Category_A'].astype(int)
df['Product Category_B'] = df['Product Category_B'].astype(int)
df['Product Category_C'] = df['Product Category_C'].astype(int)

# عرض البيانات باش نتأكد
print(df.head())

# تقسيم البيانات
X = df[['Month', 'Profit', 'Product Category_A', 'Product Category_B', 'Product Category_C']]
y = df['Sales']

# تقسيم البيانات لتدريب واختبار
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب الموديل
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# التنبؤ
y_pred = model.predict(X_test)

# عرض النتائج
from sklearn.metrics import mean_squared_error
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')



from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

