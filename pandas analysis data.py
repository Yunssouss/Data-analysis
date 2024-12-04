import pandas as bd

# قراءة ملف CSV
df = bd.read_csv(" data.csv ")

# عرض أول 5 أسطر من البيانات
print(df.head())

# تنقية البيانات (مثلاً حذف القيم الخاوية)
df_cleaned = df.dropna()

# عرض بعد التنقية
print(df_cleaned.head()) 



