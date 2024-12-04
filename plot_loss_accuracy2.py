import matplotlib.pyplot as plt

# افتراض عندك تاريخ التدريب
epochs = [1, 2, 3, 4, 5]
loss = [0.5, 0.4, 0.3, 0.2, 0.1]

plt.plot(epochs, loss)
plt.title("Training Loss over Time")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show() 

import matplotlib.pyplot as plt
import seaborn as sns

epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
loss = [0.9, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15]

plt.plot(epochs, loss, marker='o')  # 'marker=o' كيعني غنضيف نقاط دائرية فكل Epoch
plt.title("Training Loss over Epochs")  # عنوان الرسم
plt.xlabel("Epochs")  # تسمية المحور X
plt.ylabel("Loss")  # تسمية المحور Y
plt.grid(True)  # نضيف شبكة للرسم باش يكون منظم
plt.show()  # عرض الرسم

sns.set(style="whitegrid")  # اختيار ستايل 'whitegrid' للرسم
sns.lineplot(x=epochs, y=loss, marker='o')

plt.title("Training Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()




accuracy = [0.6, 0.65, 0.7, 0.75, 0.78, 0.8, 0.82, 0.85, 0.88, 0.9]

plt.plot(epochs, accuracy, marker='x', color='g')  # 'marker=x' باش يكون شكل X على كل نقطة
plt.title("Training Accuracy over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()


fig, ax1 = plt.subplots()  # نحدد Axes الأول

# رسم Loss
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss", color='tab:red')  # المحور العمودي الأول خاص بالـ Loss
ax1.plot(epochs, loss, color='tab:red', marker='o')
ax1.tick_params(axis='y', labelcolor='tab:red')

# إنشاء المحور العمودي الثاني لعرض Accuracy
ax2 = ax1.twinx()  # إنشاء محور Y آخر
ax2.set_ylabel("Accuracy", color='tab:blue')  # المحور العمودي الثاني خاص بـ Accuracy
ax2.plot(epochs, accuracy, color='tab:blue', marker='x')
ax2.tick_params(axis='y', labelcolor='tab:blue')

plt.title("Loss and Accuracy over Epochs")
fig.tight_layout()  # تنظيم الرسم
plt.show()







