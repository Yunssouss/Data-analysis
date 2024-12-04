import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# إنشاء البيانات الفلاحية لمنتوج الفلفلة من 2001 حتى 2024
years = list(range(2001, 2025))
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
product_categories = ['Green Pepper', 'Red Pepper', 'Sweet Pepper']

data = []
for year in years:
    for month in months:
        for category in product_categories:
            sales = np.random.randint(100, 300)  # مبيعات عشوائية بين 100 و 300
            profit = np.random.randint(50, 150)  # أرباح عشوائية بين 50 و 150
            data.append([f"{month} {year}", sales, profit, category])

df = pd.DataFrame(data, columns=['month', 'sales', 'profit', 'product_category'])

# تحويل الشهور إلى أرقام
df['month'] = pd.to_datetime(df['month'], format='%B %Y').dt.month

# المتغيرات المستقلة (features)
X = df[['month', 'sales', 'profit']]

# المتغير التابع (فئة المنتج)
y = df['product_category'].apply(lambda x: 0 if x == 'Green Pepper' else (1 if x == 'Red Pepper' else 2))

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تحسين Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_classifier.fit(X_train, y_train)

# توقع الفئات باستخدام بيانات الاختبار
rf_predictions = rf_classifier.predict(X_test)

# حساب الدقة
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f'Random Forest Classifier Accuracy بعد التحسين: {rf_accuracy}')

# تحسين Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
gb_classifier.fit(X_train, y_train)

# توقع الفئات باستخدام بيانات الاختبار
gb_predictions = gb_classifier.predict(X_test)

# حساب الدقة
gb_accuracy = accuracy_score(y_test, gb_predictions)
print(f'Gradient Boosting Classifier Accuracy بعد التحسين: {gb_accuracy}')


# توقع الفئات للأشهر القادمة (أكتوبر، نونبر، دجنبر 2024)
future_data = pd.DataFrame({
    'month': [10, 11, 12],  # شهور المستقبل
    'sales': [250, 260, 270],  # مبيعات متوقعة
    'profit': [150, 160, 170]  # أرباح متوقعة
})

# توقع الفئات باستخدام Random Forest Classifier
future_predictions_rf = rf_classifier.predict(future_data)
print(f'توقع الفئات باستخدام Random Forest Classifier: {future_predictions_rf}')

# توقع الفئات باستخدام Gradient Boosting Classifier
future_predictions_gb = gb_classifier.predict(future_data)
print(f'توقع الفئات باستخدام Gradient Boosting Classifier: {future_predictions_gb}')

from sklearn.model_selection import GridSearchCV

# GridSearchCV for Random Forest
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_grid_search = GridSearchCV(estimator=rf_classifier, param_grid=rf_param_grid, cv=5, n_jobs=-1, verbose=2)
rf_grid_search.fit(X_train, y_train)
best_rf = rf_grid_search.best_estimator_
print(f"Best Random Forest Params: {rf_grid_search.best_params_}")

# GridSearchCV for Gradient Boosting
gb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.05]
}
gb_grid_search = GridSearchCV(estimator=gb_classifier, param_grid=gb_param_grid, cv=5, n_jobs=-1, verbose=2)
gb_grid_search.fit(X_train, y_train)
best_gb = gb_grid_search.best_estimator_
print(f"Best Gradient Boosting Params: {gb_grid_search.best_params_}")

from imblearn.over_sampling import SMOTE

# توازن الفئات باستخدام SMOTE
sm = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = sm.fit_resample(X_train, y_train)







import RPi.GPIO as GPIO
import time

# إعداد المنافذ
GPIO.setmode(GPIO.BCM)
PIN_Relay = 18  # المنفذ المتحكم في المضخة
PIN_Sensor = 23  # المنفذ المتحكم في مستشعر الرطوبة

GPIO.setup(PIN_Relay, GPIO.OUT)
GPIO.setup(PIN_Sensor, GPIO.IN)

# وظيفة للتحكم في المضخة
def control_pump(sensor_value):
    if sensor_value == 0:  # التربة جافة
        GPIO.output(PIN_Relay, GPIO.HIGH)  # شغل المضخة
        print("التربة جافة - تشغيل المضخة")
    else:
        GPIO.output(PIN_Relay, GPIO.LOW)  # وقف المضخة
        print("التربة رطبة - وقف المضخة")

# حلقة مراقبة
try:
    while True:
        soil_moisture = GPIO.input(PIN_Sensor)
        control_pump(soil_moisture)
        time.sleep(10)  # كل 10 ثواني يتفقد الرطوبة
except KeyboardInterrupt:
    GPIO.cleanup()
