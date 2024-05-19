'''
Model B (Feature Extractor) hyper-parameter tuning / Optuna
Best parameters found:  {'hidden_size': 64, 'dropout_prob': 0.002358679520511532, 'learning_rate': 0.00606906057763516, 'weight_decay': 2.3101204300199962e-06}
Best score found:  0.9976958632469177
'''


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import hamming_loss
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import optuna


# Define the PyTorch model
class FeatureExtractor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout_prob=0.5):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


# Define the objective function for Optuna
def objective(trial):
    global X_train_B, y_train_B, X_test_B, y_test_B

    # Convert DataFrame to Tensor
    X_train_B_tensor = torch.tensor(X_train_B.values, dtype=torch.float32)
    y_train_B_tensor = torch.tensor(y_train_B.values, dtype=torch.float32)
    X_test_B_tensor = torch.tensor(X_test_B.values, dtype=torch.float32)
    y_test_B_tensor = torch.tensor(y_test_B.values, dtype=torch.float32)

    # Define the model
    model = FeatureExtractor(input_size=3, output_size=7,
                             hidden_size=trial.suggest_categorical('hidden_size', [32, 64, 128]),
                             dropout_prob=trial.suggest_uniform('dropout_prob', 0.0, 0.5))

    # Define the optimizer and learning rate
    optimizer = optim.Adam(model.parameters(),
                           lr=trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
                           weight_decay=trial.suggest_loguniform('weight_decay', 1e-6, 1e-3))

    # Define the loss function
    criterion = nn.BCEWithLogitsLoss()

    # Train the model
    epochs = 100
    batch_size = 51
    train_loader = DataLoader(TensorDataset(X_train_B_tensor, y_train_B_tensor), batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_B_tensor)
    predicted = (outputs >= 0.5).float()
    accuracy = (predicted == y_test_B_tensor).float().mean()

    return accuracy.item()


# Load dataset
large_data = pd.read_csv('data_lit_13_crystals.csv')  # Original dataset
large_data = large_data.drop_duplicates()
df_X = large_data.iloc[:, :33]  # Features
df_y = large_data.iloc[:, 33:]
df_B = pd.read_csv('las_reorder.csv')  # Small dataset B

# Prepare training and testing data for dataset B
X_B = df_B.iloc[:, :3]  # First three columns as input features
y_B = df_B.iloc[:, 3:]
X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(X_B, y_B, test_size=0.2, random_state=42)
print(X_train_B.shape)
print(y_train_B.shape)
# Run Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Print best parameters and score
print("Best parameters found: ", study.best_params)
print("Best score found: ", study.best_value)
