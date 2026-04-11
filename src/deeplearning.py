"""
COMPLETE PYTORCH DEEP LEARNING FOR CSI PLANT DISEASE DETECTION
Standalone script - no dependencies on previous analysis
"""

import numpy as np
import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set path
data_path = r'C:\Users\benhu\OneDrive\Desktop\csi_plant_data'

# Check PyTorch
print("="*80)
print("PYTORCH DEEP LEARNING FOR CSI PLANT DISEASE DETECTION")
print("="*80)
print(f"✅ PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    print("   Using CPU for training")

# ============================================================
# LOAD AND PREPROCESS DATA
# ============================================================
print("\n📂 Loading CSI data...")

def load_csi_data(filename, label, max_packets=None):
    """Load CSI data from text files"""
    full_path = f"{data_path}/{filename}"
    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    pattern = r'\[CSI #(\d+)\].*?RSSI:(-?\d+)\nAmp:\s*([\d\.\s]+)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    features = []
    for match in matches:
        amp_str = match[2].strip()
        amp_list = [float(x) for x in amp_str.split()]
        
        if len(amp_list) >= 32:  # Take first 32 subcarriers
            amp_32 = amp_list[:32]
            features.append({
                'label': label,
                'subcarriers': amp_32
            })
            
            if max_packets and len(features) >= max_packets:
                break
    
    return features

# Load datasets (balanced to 799 samples each)
print("   Loading baseline...")
baseline = load_csi_data('baseline_no_plant.txt', 0, max_packets=799)
print("   Loading healthy 30cm...")
healthy30 = load_csi_data('with_plant_30cm.txt', 1, max_packets=799)
print("   Loading healthy 45cm...")
healthy45 = load_csi_data('with_plant_45cm.txt', 1, max_packets=799)
print("   Loading diseased 30cm...")
diseased30 = load_csi_data('with_disease_plant_30cm.txt', 2, max_packets=799)
print("   Loading diseased 45cm...")
diseased45 = load_csi_data('with_disease_plant_45cm.txt', 2, max_packets=799)

# Combine all data
all_data = baseline + healthy30 + healthy45 + diseased30 + diseased45
print(f"\n✅ Total samples loaded: {len(all_data)}")

# Extract features and labels
X = np.array([d['subcarriers'] for d in all_data])
y = np.array([d['label'] for d in all_data])

print(f"   Feature shape: {X.shape}")
print(f"   Label distribution: {np.bincount(y)}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Train set: {len(X_train)} samples")
print(f"   Test set: {len(X_test)} samples")

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.LongTensor(y_train)
y_test_tensor = torch.LongTensor(y_test)

# Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ============================================================
# MODEL 1: MULTI-LAYER PERCEPTRON (MLP)
# ============================================================
print("\n" + "="*80)
print("MODEL 1: MULTI-LAYER PERCEPTRON (MLP)")
print("="*80)

class MLP(nn.Module):
    def __init__(self, input_dim=32, hidden_dims=[128, 64, 32], output_dim=3):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

mlp_model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)

print("   Training MLP...")
mlp_model.train()
for epoch in range(80):
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = mlp_model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    if (epoch + 1) % 20 == 0:
        print(f"   Epoch {epoch+1}/80, Loss: {epoch_loss/len(train_loader):.4f}")

# Evaluate
mlp_model.eval()
with torch.no_grad():
    outputs = mlp_model(X_test_tensor.to(device))
    _, predicted = torch.max(outputs, 1)
    mlp_acc = accuracy_score(y_test, predicted.cpu().numpy())
print(f"\n✅ MLP Test Accuracy: {mlp_acc*100:.2f}%")

# ============================================================
# MODEL 2: 1D CNN
# ============================================================
print("\n" + "="*80)
print("MODEL 2: 1D CONVOLUTIONAL NEURAL NETWORK")
print("="*80)

# Reshape for CNN: (batch, channels, features)
X_train_cnn = X_train.reshape(-1, 1, 32)
X_test_cnn = X_test.reshape(-1, 1, 32)

X_train_cnn_tensor = torch.FloatTensor(X_train_cnn)
X_test_cnn_tensor = torch.FloatTensor(X_test_cnn)

train_dataset_cnn = TensorDataset(X_train_cnn_tensor, y_train_tensor)
test_dataset_cnn = TensorDataset(X_test_cnn_tensor, y_test_tensor)
train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=64, shuffle=True)
test_loader_cnn = DataLoader(test_dataset_cnn, batch_size=64, shuffle=False)

class CNN1D(nn.Module):
    def __init__(self, input_channels=1, num_classes=3):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool = nn.MaxPool1d(2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

cnn_model = CNN1D().to(device)
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

print("   Training 1D CNN...")
cnn_model.train()
for epoch in range(80):
    epoch_loss = 0
    for batch_X, batch_y in train_loader_cnn:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = cnn_model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    if (epoch + 1) % 20 == 0:
        print(f"   Epoch {epoch+1}/80, Loss: {epoch_loss/len(train_loader_cnn):.4f}")

# Evaluate
cnn_model.eval()
with torch.no_grad():
    outputs = cnn_model(X_test_cnn_tensor.to(device))
    _, predicted = torch.max(outputs, 1)
    cnn_acc = accuracy_score(y_test, predicted.cpu().numpy())
print(f"\n✅ 1D CNN Test Accuracy: {cnn_acc*100:.2f}%")

# ============================================================
# MODEL 3: LSTM
# ============================================================
print("\n" + "="*80)
print("MODEL 3: LONG SHORT-TERM MEMORY (LSTM)")
print("="*80)

# Reshape for LSTM: (batch, sequence_length, features)
X_train_lstm = X_train.reshape(-1, 32, 1)
X_test_lstm = X_test.reshape(-1, 32, 1)

X_train_lstm_tensor = torch.FloatTensor(X_train_lstm)
X_test_lstm_tensor = torch.FloatTensor(X_test_lstm)

train_dataset_lstm = TensorDataset(X_train_lstm_tensor, y_train_tensor)
test_dataset_lstm = TensorDataset(X_test_lstm_tensor, y_test_tensor)
train_loader_lstm = DataLoader(train_dataset_lstm, batch_size=64, shuffle=True)
test_loader_lstm = DataLoader(test_dataset_lstm, batch_size=64, shuffle=False)

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=3):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        x = self.relu(self.fc1(last_output))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

lstm_model = LSTMClassifier().to(device)
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

print("   Training LSTM...")
lstm_model.train()
for epoch in range(80):
    epoch_loss = 0
    for batch_X, batch_y in train_loader_lstm:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = lstm_model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    if (epoch + 1) % 20 == 0:
        print(f"   Epoch {epoch+1}/80, Loss: {epoch_loss/len(train_loader_lstm):.4f}")

# Evaluate
lstm_model.eval()
with torch.no_grad():
    outputs = lstm_model(X_test_lstm_tensor.to(device))
    _, predicted = torch.max(outputs, 1)
    lstm_acc = accuracy_score(y_test, predicted.cpu().numpy())
print(f"\n✅ LSTM Test Accuracy: {lstm_acc*100:.2f}%")

# ============================================================
# MODEL 4: CNN-LSTM HYBRID
# ============================================================
print("\n" + "="*80)
print("MODEL 4: CNN-LSTM HYBRID")
print("="*80)

class CNNLSTM(nn.Module):
    def __init__(self, num_classes=3):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(64, 64, batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1)  # (batch, channels, seq) -> (batch, seq, features)
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        x = self.relu(self.fc1(last_output))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

hybrid_model = CNNLSTM().to(device)
optimizer = optim.Adam(hybrid_model.parameters(), lr=0.001)

print("   Training CNN-LSTM Hybrid...")
hybrid_model.train()
for epoch in range(80):
    epoch_loss = 0
    for batch_X, batch_y in train_loader_cnn:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = hybrid_model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    if (epoch + 1) % 20 == 0:
        print(f"   Epoch {epoch+1}/80, Loss: {epoch_loss/len(train_loader_cnn):.4f}")

# Evaluate
hybrid_model.eval()
with torch.no_grad():
    outputs = hybrid_model(X_test_cnn_tensor.to(device))
    _, predicted = torch.max(outputs, 1)
    hybrid_acc = accuracy_score(y_test, predicted.cpu().numpy())
print(f"\n✅ CNN-LSTM Test Accuracy: {hybrid_acc*100:.2f}%")

# ============================================================
# SUMMARY AND COMPARISON
# ============================================================
print("\n" + "="*80)
print("DEEP LEARNING RESULTS SUMMARY")
print("="*80)

dl_results = pd.DataFrame([
    {'Model': 'MLP', 'Test Accuracy (%)': f"{mlp_acc*100:.2f}"},
    {'Model': '1D CNN', 'Test Accuracy (%)': f"{cnn_acc*100:.2f}"},
    {'Model': 'LSTM', 'Test Accuracy (%)': f"{lstm_acc*100:.2f}"},
    {'Model': 'CNN-LSTM Hybrid', 'Test Accuracy (%)': f"{hybrid_acc*100:.2f}"}
])
print(dl_results.to_string(index=False))

# Find best model
best_dl_acc = max([mlp_acc, cnn_acc, lstm_acc, hybrid_acc])
best_dl_model = ['MLP', '1D CNN', 'LSTM', 'CNN-LSTM'][np.argmax([mlp_acc, cnn_acc, lstm_acc, hybrid_acc])]
print(f"\n🏆 Best Deep Learning Model: {best_dl_model} with {best_dl_acc*100:.2f}% accuracy")

# ============================================================
# CONFUSION MATRIX FOR BEST MODEL
# ============================================================
print("\n" + "="*80)
print("CONFUSION MATRIX (Best DL Model)")
print("="*80)

# Use best model for confusion matrix
if best_dl_model == 'MLP':
    best_model = mlp_model
    best_input = X_test_tensor
elif best_dl_model == '1D CNN':
    best_model = cnn_model
    best_input = X_test_cnn_tensor
elif best_dl_model == 'LSTM':
    best_model = lstm_model
    best_input = X_test_lstm_tensor
else:
    best_model = hybrid_model
    best_input = X_test_cnn_tensor

best_model.eval()
with torch.no_grad():
    outputs = best_model(best_input.to(device))
    _, predicted = torch.max(outputs, 1)
    y_pred = predicted.cpu().numpy()

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print("                 Predicted")
print("                 B   H   D")
print(f"Actual Baseline: {cm[0,0]:3d} {cm[0,1]:3d} {cm[0,2]:3d}")
print(f"       Healthy:  {cm[1,0]:3d} {cm[1,1]:3d} {cm[1,2]:3d}")
print(f"       Diseased: {cm[2,0]:3d} {cm[2,1]:3d} {cm[2,2]:3d}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Baseline', 'Healthy', 'Diseased']))

# ============================================================
# SAVE RESULTS
# ============================================================
dl_results.to_csv('deep_learning_results.csv', index=False)
print("\n✅ Results saved to 'deep_learning_results.csv'")

# ============================================================
# PLOT COMPARISON
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Model comparison bar chart
ax1 = axes[0]
models = ['MLP', '1D CNN', 'LSTM', 'CNN-LSTM']
accs = [mlp_acc*100, cnn_acc*100, lstm_acc*100, hybrid_acc*100]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = ax1.bar(models, accs, color=colors)
ax1.set_ylabel('Test Accuracy (%)')
ax1.set_title('Deep Learning Model Comparison')
ax1.set_ylim(70, 95)
for bar, acc in zip(bars, accs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{acc:.1f}%', ha='center', fontsize=10)

# Confusion matrix heatmap
ax2 = axes[1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=['Baseline', 'Healthy', 'Diseased'],
            yticklabels=['Baseline', 'Healthy', 'Diseased'])
ax2.set_title(f'Confusion Matrix - {best_dl_model}')
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')

plt.tight_layout()
plt.savefig('deep_learning_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Figure saved as 'deep_learning_results.png'")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*80)
print("FINAL SUMMARY FOR PAPER")
print("="*80)
print(f"""
Deep Learning Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• MLP:               {mlp_acc*100:.2f}%
• 1D CNN:            {cnn_acc*100:.2f}%
• LSTM:              {lstm_acc*100:.2f}%
• CNN-LSTM Hybrid:   {hybrid_acc*100:.2f}%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Best Deep Learning Model: {best_dl_model}
• Best Deep Learning Accuracy: {best_dl_acc*100:.2f}%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")