from pyspark.sql import SparkSession

# إنشاء جلسة عمل Spark
spark = SparkSession.builder \
    .appName("Sales Prediction Analysis") \
    .getOrCreate()

from pyspark.sql import Row

# البيانات ديال المبيعات
sales_data = [
    Row(month='January', sales=150, profit=50, product_category='A'),
    Row(month='February', sales=200, profit=80, product_category='A'),
    Row(month='March', sales=180, profit=60, product_category='B'),
    Row(month='April', sales=250, profit=100, product_category='A'),
    Row(month='May', sales=220, profit=90, product_category='B')
]

# تحويل البيانات إلى DataFrame ديال PySpark
df = spark.createDataFrame(sales_data)
df.show()

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer

# تحويل الفئات إلى قيم رقمية
indexer = StringIndexer(inputCol="product_category", outputCol="label")
df = indexer.fit(df).transform(df)

# جمع المبيعات والأرباح كـ features
assembler = VectorAssembler(inputCols=["sales", "profit"], outputCol="features")
df = assembler.transform(df)

# عرض البيانات المجهزة
df.select("features", "label").show()

from pyspark.ml.classification import LogisticRegression

# إنشاء الموديل ديال Logistic Regression
lr = LogisticRegression(featuresCol="features", labelCol="label")

# تدريب النموذج
lr_model = lr.fit(df)

# عرض coefficients ديال النموذج
print(f"Coefficients: {lr_model.coefficients}")
print(f"Intercept: {lr_model.intercept}")

# التنبؤ على نفس البيانات
predictions = lr_model.transform(df)

# عرض التوقعات
predictions.select("features", "label", "prediction").show()

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# مقياس accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy ديال الموديل: {accuracy}")


