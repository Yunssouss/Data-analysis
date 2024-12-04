import pandas as pd

class DataProcessor:
    def __init__(self, file_path):
        # هادي الميثود constructor اللي كتخزن مسار الملف كخاصية
        self.file_path = file_path
        self.data = None

    def load_data(self):
        # هادي الميثود كتقرا البيانات من CSV وتخزنها فخاصية data
        self.data = pd.read_csv(self.file_path)

    def clean_data(self):
        # ميثود تنظيف البيانات مثلا حدف القيم الناقصة
        self.data = self.data.dropna()

    def get_data(self):
        # ميثود باش نرجعو البيانات اللي تم تنظيمها
        return self.data

# استخدام الكلاس
data_processor = DataProcessor("my_data.csv")
data_processor.load_data()
data_processor.clean_data()
cleaned_data = data_processor.get_data()
print(cleaned_data.head())

import tensorflow as tf
from keras import layers

class NeuralNetworkModel:
    def __init__(self, input_shape):
        # بناء الشبكة العميقة
        self.model = tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')  # افتراض التصنيف لـ 10 فئات
        ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, x_train, y_train, epochs=10):
        # تدريب الموديل
        self.model.fit(x_train, y_train, epochs=epochs)

    def evaluate(self, x_test, y_test):
        # تقييم الموديل
        return self.model.evaluate(x_test, y_test)
 
# مثال للتدريب
        nn_model = NeuralNetworkModel(input_shape=784)
        nn_model.train(x_train, y_train, epochs=10)
        nn_model.evaluate(x_test, y_test)


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class MachineLearningModel:
    def __init__(self):
        self.model = RandomForestClassifier()

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def evaluate(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        return accuracy_score(y_test, y_pred)

# تقسيم البيانات
x_train, x_test, y_train, y_test = train_test_split( test_size=0.2)

# بناء وتدريب الموديل
ml_model = MachineLearningModel()
ml_model.train(x_train, y_train)

# تقييم الموديل
accuracy = ml_model.evaluate(x_test, y_test)
print(f"Accuracy: {accuracy}")



