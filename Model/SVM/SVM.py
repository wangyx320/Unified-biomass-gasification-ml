import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==== 定义额外指标 ====
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def nrmse(y_true, y_pred):
    return rmse(y_true, y_pred) / (np.max(y_true) - np.min(y_true))


def rmsle(y_true, y_pred):
    # 截断负值，避免 log1p 出现 invalid value
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))


# ==== 1. 读取数据 ====
df = pd.read_csv("updraft1.csv")

target_columns = ['CO', 'H2', 'CO2', 'CH4', 'LHV']
feature_columns = [col for col in df.columns if col not in target_columns]

X = df[feature_columns].values
y = df[target_columns].values

# ==== 2. KFold 设置 ====
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 保存所有样本预测结果
all_preds = np.zeros_like(y)

# 保存每折指标
fold_results = []

# ==== 3. 交叉验证 ====
for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
    print(f"\n=== 训练 Fold {fold_idx + 1}/5 ===")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 定义 SVM 多输出回归器
    svm_model = MultiOutputRegressor(SVR(kernel='rbf', C=10, epsilon=0.1))
    svm_model.fit(X_train_scaled, y_train)

    # 预测
    y_pred = svm_model.predict(X_test_scaled)
    all_preds[test_idx] = y_pred

    # 当前折指标
    fold_mae = mean_absolute_error(y_test, y_pred)
    fold_r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
    fold_rmse = rmse(y_test, y_pred)

    fold_results.append({
        'Fold': fold_idx + 1,
        'MAE': fold_mae,
        'R2': fold_r2,
        'RMSE': fold_rmse
    })

    print(f"Fold {fold_idx + 1} - MAE: {fold_mae:.4f} | R²: {fold_r2:.4f} | RMSE: {fold_rmse:.4f}")

# ==== 4. 输出交叉验证平均结果 ====
print("\n=== Cross-Validation Results (5-Fold) ===")
print(f"Mean MAE  : {np.mean([f['MAE'] for f in fold_results]):.4f}")
print(f"Mean R²   : {np.mean([f['R2'] for f in fold_results]):.4f}")
print(f"Mean RMSE : {np.mean([f['RMSE'] for f in fold_results]):.4f}")

# ==== 5. 整体评估 ====
print("\n=== Evaluation on Full Dataset ===")
mae = mean_absolute_error(y, all_preds)
mse = mean_squared_error(y, all_preds)
r2 = r2_score(y, all_preds, multioutput='uniform_average')

print("MAE  :", mae)
print("MSE  :", mse)
print("R²   :", r2)
print("RMSE :", rmse(y, all_preds))
print("NRMSE:", nrmse(y, all_preds))
print("RMSLE:", rmsle(y, all_preds))

# ==== 6. 各目标变量指标 ====
print("\n=== Per-Target Evaluation ===")
for i, col in enumerate(target_columns):
    mae_i = mean_absolute_error(y[:, i], all_preds[:, i])
    r2_i = r2_score(y[:, i], all_preds[:, i])
    rmse_i = rmse(y[:, i], all_preds[:, i])
    print(
        f"{col} - MAE: {mae_i:.3f} | R²: {r2_i:.3f} | RMSE: {rmse_i:.3f} | NRMSE: {nrmse(y[:, i], all_preds[:, i]):.3f} | RMSLE: {rmsle(y[:, i], all_preds[:, i]):.3f}")

# ==== 7. 可视化 ====
plt.figure(figsize=(15, 10))
for i, col in enumerate(target_columns):
    plt.subplot(3, 2, i + 1)
    plt.plot(y[:, i], label='Actual', linestyle='-', marker='o', markersize=4)
    plt.plot(all_preds[:, i], label='Predicted', linestyle='--', marker='x', markersize=4)
    plt.title(f'SVM Prediction - {col}')
    plt.xlabel('Sample Index')
    plt.ylabel(col)
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig("svm_updraft1_prediction_comparison.pdf", dpi=300, format='pdf')
plt.show()

# ==== 8. 导出真实值与预测值对比 ====
comparison_df = pd.DataFrame()
for i, col in enumerate(target_columns):
    comparison_df[f"{col}_Actual"] = y[:, i]
    comparison_df[f"{col}_Predicted"] = all_preds[:, i]

# 添加原始特征列
for j, col in enumerate(feature_columns):
    comparison_df[col] = X[:, j]

comparison_df.to_csv("new_svm_updraft_Predictions.csv", index=False)
print("\n已保存完整预测结果至 'svm_updraft_Predictions.csv'")
