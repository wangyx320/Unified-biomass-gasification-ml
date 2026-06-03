import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
# 计算RMSE, NRMSE, RMSLE
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def nrmse(y_true, y_pred):
    return rmse(y_true, y_pred) / (np.max(y_true) - np.min(y_true))

def rmsle(y_true, y_pred):
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))
# 读取数据
df = pd.read_csv("20250604.csv")
# df=df[325:472]

# 设定目标列和特征列
target_columns = ['CO', 'H2', 'CO2', 'CH4','LHV']
# feature_columns = [col for col in df.columns if col not in target_columns]
feature_columns = ["Time(min)","Downdraft","Updraft air","Feeding rate","Oxygen equivalence ratio","Target bed-height",
                   "50mm","100mm","200mm","300mm","400mm","500mm","600mm","700mm","800mm","900mm","950mm"]
# feature_columns = ["Downdraft air","Updraft air","Feeding rate","oxygen equivalence ratio","Target bed-height","Time","600mm:"]
X = df[feature_columns]
y = df[target_columns]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# 128, 64, 32, 16
#100,50,25
# 创建并训练优化后的 MLP 模型
mlp_model = MLPRegressor(
    hidden_layer_sizes=(128,64,32),
    activation='relu',
    alpha=1e-3,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    batch_size=32,
    max_iter=2000,
    early_stopping=False,
    validation_fraction=0.1,
    random_state=42
)
mlp_model.fit(X_train_scaled, y_train)

# 预测
y_pred_mlp = mlp_model.predict(X_test_scaled)

# 输出每个指标
print("\n=== MLP Evaluation ===")
mae = mean_absolute_error(y_test, y_pred_mlp)
mse = mean_squared_error(y_test, y_pred_mlp)
r2 = r2_score(y_test, y_pred_mlp)
print("MAE  :", mae)
print("MSE  :", mse)
print("R²   :", r2)
print("RMSE :", rmse(y_test, y_pred_mlp))
print("NRMSE:", nrmse(y_test, y_pred_mlp))
print("RMSLE:", rmsle(y_test, y_pred_mlp))

# 为每个目标变量分别输出指标
print("\n=== Per-Target Evaluation ===")
for i, col in enumerate(target_columns):
    mae = mean_absolute_error(y_test[col], y_pred_mlp[:, i])
    r2 = r2_score(y_test[col], y_pred_mlp[:, i])
    print(f"{col} - MAE: {mae:.3f} | R²: {r2:.3f} | RMSE: {rmse(y_test[col], y_pred_mlp[:, i]):.3f} | NRMSE: {nrmse(y_test[col], y_pred_mlp[:, i]):.3f} | RMSLE: {rmsle(y_test[col], y_pred_mlp[:, i]):.3f}")

# 可视化并保存为 PDF（300 DPI）
plt.figure(figsize=(12, 6))
for i, col in enumerate(target_columns):
    plt.subplot(2, 3, i + 1)
    plt.plot(y_test[col].values, label='Actual', linestyle='-', marker='o')
    plt.plot(y_pred_mlp[:, i], label='Predicted', linestyle='--', marker='x')
    plt.title(f'MLP Prediction - {col}')
    plt.xlabel('Sample Index')
    plt.ylabel(col)
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig("total_prediction_comparison.pdf", dpi=300, format='pdf')
plt.show()


# === 导出真实值与预测值对比 ===
# 创建一个新的DataFrame用于保存真实值和预测值
comparison_df = pd.DataFrame()

for i, col in enumerate(target_columns):
    comparison_df[f"{col}_Actual"] = y_test[col].values
    comparison_df[f"{col}_Predicted"] = y_pred_mlp[:, i]

# 保存为CSV文件
comparison_df.to_csv("total_MLP_Prediction_vs_Actual.csv", index=False)
print("\n已保存真实值与预测值对比数据至 'total_MLP_Prediction_vs_Actual.csv'")
