import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# تحديث البيانات بإضافة أنواع السماد المستعملة فأكادير
data = {
    'soil_type': ['رملية', 'طينية', 'طينية', 'رملية'],
    'temperature': [18, 22, 17, 25],
    'humidity': [55, 60, 50, 65],
    'rainfall': [120, 80, 150, 90],
    'wind_speed': [10, 15, 8, 12],
    'ph': [6.5, 7.2, 6.0, 7.0],
    'N': [80, 90, 70, 85],
    'P': [50, 60, 40, 55],
    'K': [200, 180, 210, 190],
    'fertilizer_N': [15, 20, 10, 30],
    'fertilizer_P': [15, 30, 10, 10],
    'fertilizer_K': [15, 10, 10, 20],
    'yield': [12.5, 13.0, 11.8, 12.9],
    'water_needs': [500, 450, 550, 480]  # هنا خاصك تحط البيانات ديال الماء
}

df = pd.DataFrame(data)


# إعادة تدريب النموذج مع البيانات الجديدة
X = df[['temperature', 'humidity', 'rainfall', 'wind_speed', 'ph', 'N', 'P', 'K', 'fertilizer_N', 'fertilizer_P', 'fertilizer_K']]
y = df['yield']

# تقسيم البيانات بين التدريب والاختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب النموذج مجددا
model = LinearRegression()
model.fit(X_train, y_train)

# توقع الإنتاج مع البيانات الجديدة
y_pred = model.predict(X_test)

# حساب الخطأ MSE
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse:.2f}')

print(df.columns)



# تأكد أن جميع المتغيرات اللي دربنا عليها الموديل راه موجودة فالتوقعات
new_data = {
    'temperature': [22],  # درجة الحرارة فشهر محدد
    'humidity': [60],      # الرطوبة
    'rainfall': [90],      # كمية الأمطار
    'wind_speed': [12],    # سرعة الرياح
    'ph': [6.5],           # الـ pH ديال التربة
    'N': [80],             # النيتروجين الموجود فالتربة
    'P': [50],             # الفوسفور فالتربة
    'K': [190],            # البوتاسيوم
    'fertilizer_N': [15],  # نسبة النيتروجين فخلطة السماد
    'fertilizer_P': [30],  # نسبة الفوسفور
    'fertilizer_K': [10],  # نسبة البوتاسيوم
}

# تحويل البيانات ل DataFrame
new_df = pd.DataFrame(new_data)



predicted_yield = model.predict(new_df[['temperature', 'humidity', 'rainfall', 'wind_speed', 'ph', 'N', 'P', 'K', 'fertilizer_N', 'fertilizer_P', 'fertilizer_K']])

print(f'توقع الإنتاج: {predicted_yield[0]:.2f} طن/هكتار')

# تدريب موديل خاص بالسماد
X_fertilizer = df[['fertilizer_N', 'fertilizer_P', 'fertilizer_K']]
y_fertilizer_N = df['N']
y_fertilizer_P = df['P']
y_fertilizer_K = df['K']

model_fertilizer_N = LinearRegression().fit(X_fertilizer, y_fertilizer_N)
model_fertilizer_P = LinearRegression().fit(X_fertilizer, y_fertilizer_P)
model_fertilizer_K = LinearRegression().fit(X_fertilizer, y_fertilizer_K)

# دابا غادي نتأكد أن الموديل مدرب على هاد المتغيرات 
# إذا كان الموديل مدرب غير على بعض المتغيرات، خصنا نديروهم بوحدهم
# مثلا إذا كان دربنا الموديل على ['temperature', 'humidity', 'rainfall', 'wind_speed', 'ph']
# نستعمل غير هاد المتغيرات فالتوقع:

# خاصك تكون عندك هاد المتغيرات معرفين من قبل
X_train_water = df[['temperature', 'humidity', 'rainfall', 'wind_speed', 'ph']]  # المتغيرات اللي غادي تستعمل فالتدريب
y_train_water = df['water_needs']  # الاحتياجات الفعلية من الماء (target)

water_model = LinearRegression().fit(X_train_water, y_train_water)


# الاحتياجات من الماء
# تدريب الموديل الخاص بالاحتياجات من الماء
water_model = LinearRegression().fit(X_train_water, y_train_water)

# دابا غادي تتوقع الاحتياجات من الماء باستعمال الموديل اللي دربتي
predicted_water_needs = water_model.predict(new_df[['temperature', 'humidity', 'rainfall', 'wind_speed', 'ph',]])
print(f'الاحتياجات من الماء: {predicted_water_needs[0]:.2f} لتر/هكتار')


# التوقعات الخاصة بالسماد
predicted_fertilizer_N = model_fertilizer_N.predict(new_df[['fertilizer_N', 'fertilizer_P', 'fertilizer_K']])
predicted_fertilizer_P = model_fertilizer_P.predict(new_df[['fertilizer_N', 'fertilizer_P', 'fertilizer_K']])
predicted_fertilizer_K = model_fertilizer_K.predict(new_df[['fertilizer_N', 'fertilizer_P', 'fertilizer_K']])

print(f'الاحتياجات من النيتروجين (N): {predicted_fertilizer_N[0]:.2f} كغ/هكتار')
print(f'الاحتياجات من الفوسفور (P): {predicted_fertilizer_P[0]:.2f} كغ/هكتار')
print(f'الاحتياجات من البوتاسيوم (K): {predicted_fertilizer_K[0]:.2f} كغ/هكتار')





