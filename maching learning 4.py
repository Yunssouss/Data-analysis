import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
print(" here ")
# إنشاء Spark session
spark = SparkSession.builder.appName("SalesPrediction").getOrCreate()

# توليد البيانات
dates = pd.date_range(start='2015-01-01', end='2023-12-31', freq='M')
products = ['مواد غذائية', 'إلكترونيات', 'ملابس', 'أدوات منزلية']
data = []
print(" here ")
for date in dates:
    for product in products:
        quantity = np.random.randint(50, 200)
        sales = quantity * np.random.uniform(20, 100)
        data.append([date, product, quantity, sales])

df = pd.DataFrame(data, columns=['date', 'product', 'quantity_sold', 'sales_amount'])
print("ملخص البيانات:\n", df.describe())
print(" here ")
# تحويل البيانات لـ Spark DataFrame
df_spark = spark.createDataFrame(df)

# استكشاف البيانات ورسمها
df_spark.show(5)
sns.lineplot(data=df, x='date', y='sales_amount', hue='product')
plt.show()

# التنبؤ باستعمال LightGBM
X = df[['quantity_sold']]
y = df['sales_amount']
model = LGBMRegressor()
model.fit(X, y)

# التقييم
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
print("MSE:", mse)
print("MAE:", mae)
  




  