import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# البيانات الأصلية
data = {
    'season': ['spring', 'summer', 'fall', 'winter', 'spring', 'summer'],
    'soil_type': ['clay', 'sandy', 'sandy', 'clay', 'clay', 'sandy'],
    'temperature': [20, 25, 22, 15, 18, 24],
    'rainfall': [250, 200, 300, 150, 180, 220],
    'humidity': [65, 70, 75, 60, 68, 72],
    'wind_speed': [10, 12, 9, 11, 10, 13],
    'ph': [6.5, 7.0, 6.8, 6.2, 6.5, 7.1],
    'N': [100, 120, 110, 90, 95, 130],
    'P': [60, 80, 70, 50, 55, 85],
    'K': [300, 350, 320, 280, 290, 360],
    'water': [700, 600, 650, 800, 750, 620],
    'fertilizer_N': [150, 170, 160, 120, 130, 180],
    'fertilizer_P': [60, 70, 65, 50, 55, 75],
    'fertilizer_K': [180, 200, 190, 160, 170, 210],
    'pepper_yield': [15, 12, 14, 17, 16, 13]
}

# تحويل البيانات إلى DataFrame
df = pd.DataFrame(data)

# ترميز الفصول ونوع التربة
encoder = LabelEncoder()
df['season'] = encoder.fit_transform(df['season'])
df['soil_type'] = encoder.fit_transform(df['soil_type'])

# تحديد المتغيرات المستقلة والتابعة
X = df.drop('pepper_yield', axis=1)
y = df['pepper_yield']

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# إنشاء نموذج XGBoost
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.05)

# تدريب النموذج
model.fit(X_train, y_train)


import pandas as pd
from sklearn.metrics import mean_squared_error

# البيانات الجديدة بعد تخفيض النسب ديال السماد والرطوبة والماء
new_data = {
    'season': [3],  # الخريف
    'soil_type': [1],  # تربة رملية
    'temperature': [20],  # درجة الحرارة فشهر 10
    'rainfall': [180],  # الساقطات المطرية، نقصناها شوية
    'humidity': [60],  # الرطوبة نقصناها شوية
    'wind_speed': [11],  # الرياح
    'ph': [6.5],  # حموضة التربة
    'N': [100],  # النيتروجين، نقصنا القيمة
    'P': [60],  # الفوسفور، نقصنا القيمة
    'K': [300],  # البوتاسيوم، نقصنا شوية
    'water': [550],  # كمية الماء نقصناها
    'fertilizer_N': [120],  # سماد النيتروجين نقصناه
    'fertilizer_P': [50],  # سماد الفوسفور نقصناه
    'fertilizer_K': [160]  # سماد البوتاسيوم نقصناه
}

# تحويل البيانات الجديدة إلى DataFrame
new_df = pd.DataFrame(new_data)

# تنبؤ الإنتاج بعد تعديل المعطيات
predicted_yield = model.predict(new_df)
print(f'الإنتاج المتنبأ به فشهر 10 بعد التعديلات هو: {predicted_yield[0]:.2f} طن/هكتار')

# حساب متوسط الخطأ التربيعي (MSE) على البيانات الاختبارية
y_pred_test = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred_test)
print(f'MSE ديال النموذج هو: {mse:.2f}')


