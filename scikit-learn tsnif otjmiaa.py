from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# نحملو البيانات
iris = load_iris()
X = iris.data
y = iris.target

# نقسمو البيانات للتدريب والاختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# نصاوب الموديل ديال Random Forest
model = RandomForestClassifier()
model.fit(X_train, y_train)

# التقييم
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy}")