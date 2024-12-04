import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# إنشاء إطار بيانات ديال المعطيات
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

# تحويل النصوص إلى أرقام
encoder = LabelEncoder()
df['season'] = encoder.fit_transform(df['season'])
df['soil_type'] = encoder.fit_transform(df['soil_type'])

# تحديد المتغيرات المستقلة والتابعة
X = df.drop('pepper_yield', axis=1)
y = df['pepper_yield']

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# إنشاء وتدريب النماذج
models = {
    "XGBoost": XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.05),
    "RandomForest": RandomForestRegressor(n_estimators=100),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.05),
    "LightGBM": LGBMRegressor(n_estimators=100, learning_rate=0.05)
}

# تدريب وتقييم النماذج
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'{name} Mean Squared Error: {mse:.2f}')

# مثال تنبؤ باستخدام LightGBM
new_data = {
    'season': [1],  # الصيف
    'soil_type': [1],  # رملية
    'temperature': [24],
    'rainfall': [220],
    'humidity': [72],
    'wind_speed': [13],
    'ph': [7.1],
    'N': [130],
    'P': [85],
    'K': [360],
    'water': [620],
    'fertilizer_N': [180],
    'fertilizer_P': [75],
    'fertilizer_K': [210]
}
new_df = pd.DataFrame(new_data)

# تنبؤ باستخدام LightGBM
predicted_yield = models["LightGBM"].predict(new_df)
print(f'The predicted pepper yield using LightGBM is: {predicted_yield[0]:.2f} tons per hectare')
