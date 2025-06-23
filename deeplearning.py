import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import re
from difflib import get_close_matches
from nltk.stem import WordNetLemmatizer
import nltk
import warnings
warnings.filterwarnings("ignore")

nltk.download('wordnet')
nltk.download('omw-1.4')

print("📥 Loading binary-encoded dataset...")
df = pd.read_csv("/content/df.csv")
df.dropna(inplace=True)

X = df.drop(columns=["diseases"])
y = df["diseases"]

labelEncoder = LabelEncoder()
y_encoded = labelEncoder.fit_transform(y)

print(f"✅ Preprocessing done. 🔢 Features: {X.shape[1]}, Samples: {X.shape[0]}")

X_train, X_test, y_train, y_test = train_test_split(
    X.values, y_encoded, test_size=0.2, random_state=42
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self, inputSize, numClasses):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(inputSize, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, numClasses)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

print("🔥 Training PyTorch model...")
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
trainDataset = TensorDataset(X_train_tensor, y_train_tensor)
trainLoader = DataLoader(trainDataset, batch_size=256, shuffle=True)

modelTorch = Net(X.shape[1], len(np.unique(y_encoded))).to(device)
classWeights = torch.tensor(
    1.0 / np.bincount(y_train), dtype=torch.float32
).to(device)
loss_fn = nn.CrossEntropyLoss(weight=classWeights)
optimizer = optim.Adam(modelTorch.parameters(), lr=0.001)

modelTorch.train()
for epoch in range(50):
    epochLoss = 0.0
    for inputs, labels in trainLoader:
        optimizer.zero_grad()
        outputs = modelTorch(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        epochLoss += loss.item()
    print(f"🧠 Torch Epoch {epoch+1} Loss: {epochLoss:.4f}")

print("🔥 Training TensorFlow model...")
modelTF = Sequential([
    Dense(256, input_shape=(X.shape[1],), activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(np.unique(y_encoded)), activation='softmax')
])
modelTF.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

modelTF.fit(
    X_train, to_categorical(y_train),
    validation_data=(X_test, to_categorical(y_test)),
    epochs=50, batch_size=256, verbose=1
)

modelTF.save("tensorflow_model.h5")
torch.save(modelTorch.state_dict(), "torch_model.pth")
print("✅ Models saved.")

lemmatizer = WordNetLemmatizer()

def cleanAndLemmatize(text):
    return [lemmatizer.lemmatize(w) for w in re.findall(r'\b[a-z]+\b', text.lower())]

def matchSymptoms(userInput, symptomList, threshold=0.8):
    words = cleanAndLemmatize(userInput)
    matchedSymptoms = set()
    for word in words:
        close = get_close_matches(word, symptomList, n=1, cutoff=threshold)
        if close:
            matchedSymptoms.add(close[0])
    return list(matchedSymptoms)

def predictDisease(msg):
    inputVector = np.zeros(X.shape[1])
    matched = matchSymptoms(msg, X.columns.tolist())
    for symptom in matched:
        if symptom in X.columns:
            idx = X.columns.get_loc(symptom)
            inputVector[idx] = 1.0

    inputTensor = torch.tensor(inputVector, dtype=torch.float32).to(device)
    modelTorch.eval()
    with torch.no_grad():
        outputTorch = modelTorch(inputTensor)
        predTorch = torch.softmax(outputTorch, dim=0).cpu().numpy()

    outputTF = modelTF.predict(np.array([inputVector]), verbose=0)[0]

    finalProbs = (predTorch + outputTF) / 2
    topIndices = finalProbs.argsort()[::-1][:3]
    topDiseases = [(labelEncoder.inverse_transform([i])[0], finalProbs[i]*100) for i in topIndices]

    return matched, topDiseases
