import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim

# Caricamento del dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
data = pd.read_csv(url, header=None, names=col_names)

# Calcolo della correlazione tra due attributi
corr = np.corrcoef(data['sepal_length'], data['petal_length'])
print('Correlazione tra sepal_length e petal_length:', corr[0][1])

# Preparazione dei dati
X = data.iloc[:, :-1].values
y = pd.factorize(data['class'])[0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Implementazione del MLP con Sklearn
model_sklearn = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, solver='sgd', random_state=0)
model_sklearn.fit(X_train, y_train)

# Implementazione del MLP con PyTorch
input_size = X_train.shape[1]
hidden_size = 10
output_size = len(np.unique(y_train))
n_epochs = 1000
batch_size = 10

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

mlp_pt = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(mlp_pt.parameters(), lr=0.1)

inputs = torch.from_numpy(X_train)
targets = torch.from_numpy(y_train)

for epoch in range(n_epochs):
    optimizer.zero_grad()
    outputs = mlp_pt(inputs.float())
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

# Valutazione dei modelli sul test set
# Implementazione del MLP con Sklearn
y_pred_sklearn = model_sklearn.predict(X_test)
print('MLP con Sklearn')
print(classification_report(y_test, y_pred_sklearn))
print(confusion_matrix(y_test, y_pred_sklearn))

# Implementazione del MLP con PyTorch
mlp_pt.eval()
with torch.no_grad():
    inputs = torch.from_numpy(X_test)
    outputs = mlp_pt(inputs.float())
    _, y_pred_pt = torch.max(outputs, 1)

print('MLP con PyTorch')
print(classification_report(y_test, y_pred_pt))
print(confusion_matrix(y_test, y_pred_pt))
