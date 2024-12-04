# مثال على إنشاء بيانات خيالية باستعمال pandas
import pandas as pd

# إنشاء جدول البيانات
data = {
    'Month': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
    'Sales': [150, 200, 180, 250, 220, 210, 190, 230, 210, 240, 280, 300],
    'Product Category': ['A', 'A', 'B', 'A', 'B', 'B', 'C', 'C', 'A', 'B', 'C', 'A'],
    'Profit': [50, 80, 60, 100, 90, 85, 75, 95, 80, 110, 130, 150]
}

df = pd.DataFrame(data)
print(df)



# تحليلات بسيطة
monthly_sales = df.groupby('Month')['Sales'].sum()
category_profit = df.groupby('Product Category')['Profit'].sum()

print(monthly_sales)
print(category_profit)

import matplotlib.pyplot as plt
import seaborn as sns

# رسم بياني للمبيعات الشهرية
plt.figure(figsize=(10, 6))
sns.barplot(x='Month', y='Sales', data=df)
plt.title('Monthly Sales')
plt.xticks(rotation=45)
plt.show()

# رسم بياني للأرباح حسب الفئات
plt.figure(figsize=(8, 5))
sns.barplot(x='Product Category', y='Profit', data=df)
plt.title('Profit by Product Category')
plt.show()

# الرسم البياني للارتباط بين المبيعات والأرباح
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Sales', y='Profit', hue='Product Category', data=df)
plt.title('Sales vs Profit')
plt.show()

import sqlite3

# إنشاء أو الاتصال بقاعدة البيانات
conn = sqlite3.connect('monthly_sales.db')
cursor = conn.cursor()

# حذف الجدول إذا كان موجود من قبل
cursor.execute('DROP TABLE IF EXISTS sales')

# إنشاء جدول جديد للمبيعات مع قيد فريد على الشهر
cursor.execute('''
    CREATE TABLE IF NOT EXISTS sales (
        id INTEGER PRIMARY KEY,
        month TEXT,
        sales INTEGER,
        profit INTEGER,
        product_category TEXT
    )
''')

# تأكيد التغييرات
conn.commit()

# البيانات الشهرية
monthly_sales = [
    ('January', 150, 50, 'A'),
    ('February', 200, 80, 'A'),
    ('March', 180, 60, 'B'),
    ('April', 250, 100, 'A'),
    ('May', 220, 90, 'B')
]

# إدخال البيانات الجديدة مرة وحدة فقط
cursor.executemany('''
    INSERT INTO sales (month, sales, profit, product_category)
    VALUES (?, ?, ?, ?)
''', monthly_sales)

# تأكيد التغييرات
conn.commit()

# استرجاع جميع البيانات بدون تكرار
cursor.execute('SELECT * FROM sales')
rows = cursor.fetchall()

# عرض البيانات
print("البيانات المدخلة:")
for row in rows:
    print(row)

# حساب إجمالي المبيعات
cursor.execute('SELECT SUM(sales) FROM sales')
total_sales = cursor.fetchone()[0]

print("\nإجمالي المبيعات:", total_sales)

# حساب متوسط الأرباح
cursor.execute('SELECT AVG(profit) FROM sales')
average_profit = cursor.fetchone()[0]

print("متوسط الأرباح:", average_profit)

# تصنيف المبيعات حسب فئة المنتج
cursor.execute('SELECT product_category, SUM(sales) FROM sales GROUP BY product_category')
category_sales = cursor.fetchall()

print("\nمبيعات حسب فئة المنتج:")
for row in category_sales:
    print(f"الفئة {row[0]} مبيعاتها: {row[1]}")

# إغلاق الاتصال بقاعدة البيانات
conn.close()





#نظم العرض: تأكد أن كل خطوة مشروحة بوضوح، من تحليل البيانات حتى التصور، وتقدر دير العرض في ملف Jupyter Notebook باش الناس يشوفو الكود والنتائج في نفس الوقت.
#شرح النتائج: من بعد التصور، خصك تشرح شنو اللي استنتجتي من التحليل. مثلا، تقدر تقول "لاحظنا أن مبيعات شهر نوفمبر هي الأكثر ارتفاعًا بسبب زيادة الطلب على الفئة A".
#الثقة #في النفس: كون واثق من العرض ديالك. حاول تحضر شوية حول تحليل البيانات وتصوره باش تكون جاهز تجاوب على الأسئلة. كون متأكد أن المشروع اللي خدمت عليه يقدر يعطيهم فكرة واضحة على مهاراتك
