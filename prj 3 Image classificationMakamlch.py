from tensorflow.keras.datasets import cifar10

# تحميل البيانات
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# طبع البيانات باش تفهم شنو فيها
print(f"Training data shape: {X_train.shape}, Labels: {y_train.shape}")
print(f"Test data shape: {X_test.shape}, Labels: {y_test.shape}")
 
 # تطبيع البيانات
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# تحويل الفئات إلى صيغة واحدات صفرية (One-hot encoding)
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# بناء النموذج
model = Sequential()

# الطبقة الالتفافية الأولى
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# الطبقة الالتفافية الثانية
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# تحويل البيانات إلى 1D
model.add(Flatten())

# طبقات كاملة الربط
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))  # 10 فئات للتصنيف

# طباعة ملخص النموذج
model.summary()

# إعداد النموذج
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# تدريب النموذج
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))

# تقييم النموذج
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

from PIL import Image
import numpy as np

# تحميل الصورة وتغيير حجمها
img = Image.open("bird_17_p2.jpg")
img = img.resize((32, 32))  # تغيير الحجم باش يتوافق مع المدخلات ديال الشبكة

# تحويل الصورة إلى مصفوفة
img_array = np.array(img)

# تطبيع الصورة بحال كيفما درنا فالتدريب
img_array = img_array.astype('float32') / 255.0

# إعادة تشكيل البيانات باش تقدر الشبكة تقراها (batch_size, height, width, channels)
img_array = np.expand_dims(img_array, axis=0)

print(img_array.shape)  # التأكد من الشكل النهائي ديال الصورة

# توقع الفئة
prediction = model.predict(img_array)

# تحويل النتيجة إلى الفئة المتوقعة
predicted_class = np.argmax(prediction, axis=1)
print(f"الفئة المتوقعة: {predicted_class}")

import matplotlib.pyplot as plt
classes = ['طائرة', 'سيارة', 'عصفور', 'قطة', 'غزالة', 'كلب', 'ضفدع', 'حصان', 'سفينة', 'شاحنة']  # أصناف CIFAR-10

# عرض الصورة
plt.imshow(np.squeeze(img_array))  # np.squeeze باش نحيدو البُعد الزائد اللي زدناه
plt.title(f"الفئة المتوقعة: {classes[predicted_class[0]]}")
plt.axis('off')  # باش نحيدو المحاور
plt.show()

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# تحميل الصورة من ملف
img = Image.open("bird_17_p2.jpg")
img = img.resize((32, 32))  # تغيير الحجم

# تجهيز الصورة
img_array = np.array(img)
img_array = img_array.astype('float32') / 255.0
img_array = np.expand_dims(img_array, axis=0)

# توقع الفئة
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction, axis=1)

# عرض الصورة مع النتيجة
plt.imshow(np.squeeze(img_array))
plt.title(f"الفئة المتوقعة: {classes[predicted_class[0]]}")
plt.axis('off')
plt.show()
