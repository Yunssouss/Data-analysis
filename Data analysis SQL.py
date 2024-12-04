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

