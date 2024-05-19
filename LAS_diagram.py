"""
data from LAS diagram (dataset B)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import hamming_loss, classification_report, f1_score, precision_score
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('las_reorder.csv')
df = df.drop_duplicates()

X = df.iloc[:, :3]
y = df.iloc[:, 3:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


input_size = 3
hidden_size = 64
output_size = 7
learning_rate = 0.001
num_epochs = 100

model = NeuralNet(input_size, hidden_size)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    outputs = model(X_test)
    predicted = (outputs >= 0.5).float()
    accuracy = (predicted == y_test.unsqueeze(1)).float().mean()
    print(f'Accuracy on test set: {accuracy.item():.4f}')

hamming_distance = hamming_loss(y_test.numpy(), predicted.numpy())

print(f'Hamming Loss on test set: {hamming_distance:.4f}')


##################################################################################
#                                traditional ML
##################################################################################
import pandas as pd

df_B = pd.read_csv('las_reorder.csv')
df_B = df_B.drop_duplicates()

X_B = df_B.iloc[:, :3]
y_B = df_B.iloc[:, 3:]

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, Y_train, Y_test = train_test_split(X_B, y_B, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
cart_classifier = DecisionTreeClassifier(random_state=42)
knn_classifier = KNeighborsClassifier(n_neighbors=5)

rf_classifier.fit(X_train, Y_train)
cart_classifier.fit(X_train, Y_train)
knn_classifier.fit(X_train, Y_train)

Y_pred_rf = rf_classifier.predict(X_test)
Y_pred_cart = cart_classifier.predict(X_test)
Y_pred_knn = knn_classifier.predict(X_test)

accuracy_rf = accuracy_score(Y_test, Y_pred_rf)
accuracy_cart = accuracy_score(Y_test, Y_pred_cart)
accuracy_knn = accuracy_score(Y_test, Y_pred_knn)
print("RF accuracy", accuracy_rf)
print("CART accuracy:", accuracy_cart)
print("KNN accuracy:", accuracy_knn)

from sklearn.metrics import hamming_loss

hamming_distance_rf = hamming_loss(Y_test, Y_pred_rf)
hamming_distance_cart = hamming_loss(Y_test, Y_pred_cart)
hamming_distance_knn = hamming_loss(Y_test, Y_pred_knn)

print("RF Hamming Distance:", hamming_distance_rf)
print("CART Hamming Distance:", hamming_distance_cart)
print("KNN Hamming Distance:", hamming_distance_knn)

F1_score_rf = f1_score(Y_test, Y_pred_rf, average='micro')
F1_score_cart = f1_score(Y_test, Y_pred_cart, average='micro')
F1_score_knn = f1_score(Y_test, Y_pred_knn, average='micro')

print('RF F1 score :', F1_score_rf)
print('CART F1 score :', F1_score_cart)
print('KNN F1 score :', F1_score_knn)
