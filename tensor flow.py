import tensorflow as tf 

from tensorflow import layers, models
 
# نصاوب نموذج (Model) للتعلم العميق
model = models.Sequential()

# نضيفو لاييرات
model.add(layers.Dense(64, activation='relu', input_shape=(100,)))
model.add(layers.Dense(10, activation='softmax'))

# نحدد المعايير ديال التدريب
model.compile(optimizer='adam', loss='categorical_crossentropy')

# عرض بنية النموذج
model.summary()
