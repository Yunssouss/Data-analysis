import pandas as pd

# عينة من التعليقات
data = {
    'comment': [
        'This product is amazing, I love it!',
        'Worst purchase ever, I regret buying it.',
        'Pretty decent, but has some flaws.',
        'Absolutely fantastic! Exceeded my expectations.',
        'Not worth the money, very disappointing.'
    ]
}

df = pd.DataFrame(data)
print(df)

import re
import nltk 
from nltk.corpus import stopwords

# نحمل الكلمات الشائعة اللي بغينا نحيدو
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # نحيد الرموز الزائدة ونخلي غير الحروف
    text = re.sub(r'[^\w\s]', '', text.lower())
    # نقسم النص على شكل كلمات (tokens)
    tokens = text.split()
    # نحيد الكلمات الشائعة
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['cleaned_comment'] = df['comment'].apply(preprocess_text)
print(df)

from sklearn.model_selection import train_test_split

# تحديد التعليقات الإيجابية ب 1 و السلبية ب 0
df['sentiment'] = [1, 0, 1, 1, 0]  # التعليقات
X = df['cleaned_comment']
y = df['sentiment']

# نقسم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.optimizers import Adam

# نحمل BERT tokenizer و model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# نجهزو البيانات للـ BERT
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=128, return_tensors='tf')
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=128, return_tensors='tf')

# إعداد النموذج للتدريب
model.compile(optimizer=Adam(learning_rate=3e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# تدريب النموذج
history = model.fit(train_encodings['input_ids'], y_train, epochs=3, batch_size=16, validation_data=(test_encodings['input_ids'], y_test))

# تقييم النموذج
results = model.evaluate(test_encodings['input_ids'], y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")








