import pandas as pd

# إنشاء إطار بيانات ديال المعطيات
data = {
    'season': ['spring', 'summer', 'fall', 'winter', 'spring', 'summer'],
    'soil_type': ['clay', 'sandy', 'sandy', 'clay', 'clay', 'sandy'],
    'temperature': [20, 25, 22, 15, 18, 24],  # درجة الحرارة (متوسط سنوي)
    'rainfall': [250, 200, 300, 150, 180, 220],  # الساقطات المطرية (ملمتر)
    'humidity': [65, 70, 75, 60, 68, 72],  # الرطوبة (%)
    'wind_speed': [10, 12, 9, 11, 10, 13],  # سرعة الرياح (كم/ساعة)
    'ph': [6.5, 7.0, 6.8, 6.2, 6.5, 7.1],  # حموضة التربة (PH)
    'N': [100, 120, 110, 90, 95, 130],  # النيتروجين (mg/kg)
    'P': [60, 80, 70, 50, 55, 85],  # الفوسفور (mg/kg)
    'K': [300, 350, 320, 280, 290, 360],  # البوتاسيوم (mg/kg)
    'water': [700, 600, 650, 800, 750, 620],  # كمية الماء المستعملة (ملمتر/موسم)
    'fertilizer_N': [150, 170, 160, 120, 130, 180],  # سماد النيتروجين (كغ/هكتار)
    'fertilizer_P': [60, 70, 65, 50, 55, 75],  # سماد الفوسفور (كغ/هكتار)
    'fertilizer_K': [180, 200, 190, 160, 170, 210],  # سماد البوتاسيوم (كغ/هكتار)
    'pepper_yield': [15, 12, 14, 17, 16, 13]  # الإنتاج (طن/هكتار)
}

# تحويل البيانات إلى DataFrame
df = pd.DataFrame(data)

from sklearn.preprocessing import LabelEncoder

# تحويل النصوص إلى أرقام
encoder = LabelEncoder()
df['season'] = encoder.fit_transform(df['season'])
df['soil_type'] = encoder.fit_transform(df['soil_type'])

print(df.head())  # عرض المعطيات باش نتأكد أنها صالحة

from sklearn.model_selection import train_test_split

# تحديد المتغيرات المستقلة والتابعة
X = df.drop('pepper_yield', axis=1)  # المتغيرات المستقلة
y = df['pepper_yield']  # المتغير التابع

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


import xgboost as xgb
from sklearn.metrics import mean_squared_error

# إنشاء نموذج XGBoost
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.05)

# تدريب النموذج
model.fit(X_train, y_train)

# التنبؤ على البيانات الاختبارية
y_pred = model.predict(X_test)

# حساب الخطأ
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# تنبؤ كمية الإنتاج على البيانات الجديدة
new_data = {
    'season': [1],  # الصيف
    'soil_type': [1],  # رملية
    'temperature': [24],  # درجة الحرارة
    'rainfall': [220],  # الساقطات المطرية
    'humidity': [72],  # الرطوبة
    'wind_speed': [13],  # الرياح
    'ph': [7.1],  # حموضة التربة
    'N': [130],  # النيتروجين
    'P': [85],  # الفوسفور
    'K': [360],  # البوتاسيوم
    'water': [620],  # الماء
    'fertilizer_N': [180],  # سماد النيتروجين
    'fertilizer_P': [75],  # سماد الفوسفور
    'fertilizer_K': [210]  # سماد البوتاسيوم
}

# تحويل البيانات الجديدة إلى DataFrame
new_df = pd.DataFrame(new_data)

# تنبؤ الإنتاج
predicted_yield = model.predict(new_df)
print(f'The predicted pepper yield is: {predicted_yield[0]} tons per hectare')


# إنشاء معطيات تجريبية لمختلف الفصول مع الحفاظ على الشروط المناخية الأخرى
season_data = {
    'season': [0, 1, 2, 3],  # الفصول: ربيع، صيف، خريف، شتاء
    'soil_type': [1, 1, 1, 1],  # تربة رملية
    'temperature': [22, 24, 20, 15],  # درجات الحرارة لكل فصل
    'rainfall': [300, 220, 250, 150],  # الساقطات المطرية
    'humidity': [75, 72, 65, 60],  # الرطوبة
    'wind_speed': [9, 13, 10, 11],  # سرعة الرياح
    'ph': [6.8, 7.1, 6.5, 6.2],  # حموضة التربة
    'N': [110, 130, 100, 90],  # النيتروجين
    'P': [70, 85, 60, 50],  # الفوسفور
    'K': [320, 360, 300, 280],  # البوتاسيوم
    'water': [650, 620, 700, 800],  # كمية الماء المستعملة
    'fertilizer_N': [160, 180, 150, 120],  # سماد النيتروجين
    'fertilizer_P': [65, 75, 60, 50],  # سماد الفوسفور
    'fertilizer_K': [190, 210, 180, 160]  # سماد البوتاسيوم
}

# تحويل البيانات الجديدة إلى DataFrame
season_df = pd.DataFrame(season_data)

# تنبؤ كمية الإنتاج لكل فصل
predicted_yields = model.predict(season_df)

# عرض نتائج التنبؤ
for i, yield_ in enumerate(predicted_yields):
    print(f"الفصل {i} يعطي إنتاجية متنبأ بها: {yield_:.2f} طن/هكتار")




from sklearn.preprocessing import LabelEncoder

# ترميز الفصول
seasons = ['winter', 'spring', 'summer', 'fall']  # مثال ديال الفصول
encoder = LabelEncoder()
encoded_seasons = encoder.fit_transform(seasons)

# عرض الفصول المرتبطة بكل رقم
print(encoder.classes_)  # هاد السطر غادي يعطينا الترتيب الصحيح ديال الفصول


# معرفة أحسن فصل للزراعة بناءً على الإنتاجية
best_season_index = predicted_yields.argmax()
best_season = encoder.inverse_transform([best_season_index])[0]
print(f"أحسن وقت للزراعة هو فصل {best_season} بناءً على التنبؤ.")

