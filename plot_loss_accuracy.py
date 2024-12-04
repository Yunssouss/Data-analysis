import matplotlib.pyplot as plt

# بيانات للتدريب
epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
loss = [0.9, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15]

# رسم loss
plt.plot(epochs, loss, marker='o')
plt.title("Training Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)

# عرض الرسم
plt.show()



