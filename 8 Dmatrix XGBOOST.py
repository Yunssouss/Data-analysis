import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

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

# تحويل الفصول ونوع التربة إلى أرقام
encoder = LabelEncoder()
df['season'] = encoder.fit_transform(df['season'])
df['soil_type'] = encoder.fit_transform(df['soil_type'])

# تحديد المتغيرات المستقلة والتابعة
X = df.drop('pepper_yield', axis=1)
y = df['pepper_yield']

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تحويل البيانات إلى DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# إعدادات النموذج
params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.05
}

# تدريب النموذج
model = xgb.train(params, dtrain, num_boost_round=100)

# التنبؤ على البيانات الاختبارية
y_pred = model.predict(dtest)

# حساب MSE
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')



# تحويل البيانات إلى DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# إعدادات النموذج
params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.05
}

# تدريب النموذج باستخدام DMatrix
model = xgb.train(params, dtrain, num_boost_round=100)

# التنبؤ على البيانات الاختبارية
y_pred = model.predict(dtest)

# حساب MSE
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# تنبؤ كمية الإنتاج على بيانات جديدة
new_data = {
    'season': [1], 'soil_type': [1], 'temperature': [24], 'rainfall': [220],
    'humidity': [72], 'wind_speed': [13], 'ph': [7.1], 'N': [130],
    'P': [85], 'K': [360], 'water': [620], 'fertilizer_N': [180],
    'fertilizer_P': [75], 'fertilizer_K': [210]
}

new_df = pd.DataFrame(new_data)
dnew = xgb.DMatrix(new_df)

# التنبؤ بكمية الإنتاج
predicted_yield = model.predict(dnew)
print(f'The predicted pepper yield is: {predicted_yield[0]} tons per hectare')
