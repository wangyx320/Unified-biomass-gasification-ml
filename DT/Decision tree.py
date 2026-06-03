import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor


# === 自定义指标 ===
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def nrmse(y_true, y_pred):
    return rmse(y_true, y_pred) / (np.max(y_true) - np.min(y_true))


def rmsle(y_true, y_pred):
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))


# === 读取数据 ===
df = pd.read_csv("20250604.csv")

# 设定目标列和特征列
target_columns = ['CO', 'H2', 'CO2', 'CH4', 'LHV']
feature_columns = [col for col in df.columns if col not in target_columns]

X = df[feature_columns].values
y = df[target_columns].values

# 特征标准化（决策树不敏感，这里可选）
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 初始化KFold交叉验证（5折）
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 存储预测结果
all_preds = np.zeros_like(y)

# 存储每折的结果
fold_results = []

# === 交叉验证循环 ===
for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
    print(f"\n=== 训练Fold {fold_idx + 1}/5 ===")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 训练模型
    tree_model = DecisionTreeRegressor(random_state=42)
    tree_model.fit(X_train, y_train)

    # 预测
    y_pred = tree_model.predict(X_test)
    all_preds[test_idx] = y_pred

    # 评估指标
    fold_mae = mean_absolute_error(y_test, y_pred)
    fold_r2 = r2_score(y_test, y_pred)
    fold_rmse = rmse(y_test, y_pred)

    fold_results.append({
        "Fold": fold_idx + 1,
        "MAE": fold_mae,
        "R2": fold_r2,
        "RMSE": fold_rmse
    })

    print(f"Fold {fold_idx + 1} - MAE: {fold_mae:.4f} | R²: {fold_r2:.4f} | RMSE: {fold_rmse:.4f}")

# === 平均结果 ===
print("\n=== 交叉验证平均结果 ===")
print(f"平均MAE  : {np.mean([f['MAE'] for f in fold_results]):.4f}")
print(f"平均R²   : {np.mean([f['R2'] for f in fold_results]):.4f}")
print(f"平均RMSE : {np.mean([f['RMSE'] for f in fold_results]):.4f}")

# === 整体数据集评估 ===
print("\n=== 完整数据集评估 ===")
mae = mean_absolute_error(y, all_preds)
mse = mean_squared_error(y, all_preds)
r2 = r2_score(y, all_preds)
print("MAE  :", mae)
print("MSE  :", mse)
print("R²   :", r2)
print("RMSE :", rmse(y, all_preds))
print("NRMSE:", nrmse(y, all_preds))
print("RMSLE:", rmsle(y, all_preds))

# === 各目标变量单独评估 ===
print("\n=== 各目标变量评估 ===")
for i, col in enumerate(target_columns):
    mae = mean_absolute_error(y[:, i], all_preds[:, i])
    r2 = r2_score(y[:, i], all_preds[:, i])
    print(f"{col} - MAE: {mae:.3f} | R²: {r2:.3f} | RMSE: {rmse(y[:, i], all_preds[:, i]):.3f} | "
          f"NRMSE: {nrmse(y[:, i], all_preds[:, i]):.3f} | RMSLE: {rmsle(y[:, i], all_preds[:, i]):.3f}")

# === 可视化并保存PDF ===
plt.figure(figsize=(15, 10))
for i, col in enumerate(target_columns):
    plt.subplot(3, 2, i + 1)
    plt.plot(y[:, i], label='Actual', linestyle='-', marker='o', markersize=3)
    plt.plot(all_preds[:, i], label='Predicted', linestyle='--', marker='x', markersize=3)
    plt.title(f'Decision Tree Prediction - {col}')
    plt.xlabel('Sample Index')
    plt.ylabel(col)
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig("decision_tree_total_prediction.pdf", dpi=300, format='pdf')
plt.show()

# === 导出真实值与预测值 ===
comparison_df = pd.DataFrame()
for i, col in enumerate(target_columns):
    comparison_df[f"{col}_Actual"] = y[:, i]
    comparison_df[f"{col}_Predicted"] = all_preds[:, i]

# 添加特征列（可追溯性）
for j, col in enumerate(feature_columns):
    comparison_df[col] = X[:, j]

comparison_df.to_csv("decision_tree_total_prediction.csv", index=False)
print("\n已保存完整预测结果至 'decision_tree_total_prediction.csv'")
