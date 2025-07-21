# transformer_regression_gpu.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# -------------------- Transformer Model --------------------
class TransformerRegressor(nn.Module):
    def __init__(self, input_dim=5, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = x.unsqueeze(1)  # batch_first: (B, 1, D)
        x = self.transformer_encoder(x)
        x = self.norm(x)
        x = self.output_proj(x[:, 0])
        return x.squeeze(-1)

# -------------------- 数据加载和预处理 --------------------
def load_and_preprocess_data():
    df_bc = pd.read_excel("A不变BC都变.xlsx", engine='openpyxl')
    df_bc.columns = ['B掺杂比', 'C掺杂比', '强度']

    df_bc['B掺杂比'] = pd.to_numeric(df_bc['B掺杂比'], errors='coerce')
    df_bc['C掺杂比'] = pd.to_numeric(df_bc['C掺杂比'], errors='coerce')
    df_bc['强度'] = pd.to_numeric(df_bc['强度'], errors='coerce')
    df_bc = df_bc.dropna()

    df_bc['BxC'] = df_bc['B掺杂比'] * df_bc['C掺杂比']
    df_bc['B_squared'] = df_bc['B掺杂比'] ** 2
    df_bc['C_squared'] = df_bc['C掺杂比'] ** 2
    df_bc['log_强度'] = np.log1p(df_bc['强度'])
    return df_bc

# -------------------- 训练 Transformer 模型 --------------------
def train_transformer_model(df, epochs=200, batch_size=32, lr=1e-3):
    features = ['B掺杂比', 'C掺杂比', 'BxC', 'B_squared', 'C_squared']
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[features].values.astype('float32'))
    y = df['log_强度'].values.astype('float32')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train).to(device), torch.tensor(y_train).to(device)),
                              batch_size=batch_size, shuffle=True)

    model = TransformerRegressor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test).to(device)
        y_pred_log = model(X_test_tensor).cpu().numpy()
        y_pred = np.expm1(y_pred_log)
        y_true = np.expm1(y_test)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"Test RMSE: {rmse:.2f}, R2: {r2:.4f}")

    return model, device, scaler

# -------------------- 预测最佳配比 --------------------
def predict_optimal_ratio(model, device, scaler):
    b_vals = np.linspace(0, 0.078, 100)
    c_vals = np.linspace(0, 0.78, 100)
    bb, cc = np.meshgrid(b_vals, c_vals)

    df_grid = pd.DataFrame({
        'B掺杂比': bb.ravel(),
        'C掺杂比': cc.ravel(),
        'BxC': (bb * cc).ravel(),
        'B_squared': (bb ** 2).ravel(),
        'C_squared': (cc ** 2).ravel()
    })

    X_scaled = scaler.transform(df_grid.values.astype('float32'))

    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled).to(device)
        pred_log = model(X_tensor).cpu().numpy()
        pred_intensity = np.expm1(pred_log)

    best_idx = np.argmax(pred_intensity)
    best_b = bb.ravel()[best_idx]
    best_c = cc.ravel()[best_idx]
    best_y = pred_intensity[best_idx]

    print(f"最佳B掺杂比: {best_b:.4f}, C掺杂比: {best_c:.4f}, 强度: {best_y:.2f}")
    return bb, cc, pred_intensity.reshape(bb.shape), best_b, best_c, best_y

# -------------------- 主程序 --------------------
def main():
    df = load_and_preprocess_data()
    model, device, scaler = train_transformer_model(df)
    bb, cc, z_pred, best_b, best_c, best_y = predict_optimal_ratio(model, device, scaler)

    plt.figure(figsize=(10, 5))
    plt.contourf(bb, cc, z_pred, levels=20, cmap='viridis')
    plt.colorbar(label='预测强度')
    plt.scatter(best_b, best_c, c='red', s=100, label='最佳配比')
    plt.xlabel('B掺杂比')
    plt.ylabel('C掺杂比')
    plt.legend()
    plt.title('Transformer 预测强度热图')
    plt.tight_layout()
    plt.savefig('transformer_prediction_heatmap.png')
    plt.show()

if __name__ == "__main__":
    main()