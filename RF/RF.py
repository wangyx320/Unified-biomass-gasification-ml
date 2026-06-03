import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


# =========================================================
# 自定义评价指标
# =========================================================
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def nrmse(y_true, y_pred):
    return rmse(y_true, y_pred) / (np.max(y_true) - np.min(y_true))


def rmsle(y_true, y_pred):
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))


# =========================================================
# 读取数据
# =========================================================
df = pd.read_csv("2026.csv")

target_columns = ['CO', 'H2', 'CO2', 'CH4', 'LHV']
feature_columns = [col for col in df.columns if col not in target_columns]

X = df[feature_columns].values
y = df[target_columns].values


# =========================================================
# 【可选】特征标准化（RF一般不需要）
# =========================================================
USE_SCALER = False   # 【可调参数1】是否标准化
if USE_SCALER:
    scaler = StandardScaler()
    X = scaler.fit_transform(X)


# =========================================================
# KFold交叉验证
# =========================================================
kf = KFold(
    n_splits=5,        # 【可调参数2】折数
    shuffle=True,
    random_state=42    # 【可调参数3】随机种子
)

all_preds = np.zeros_like(y)
fold_results = []


# =========================================================
# RF模型参数（重点调这里）
# =========================================================
def build_rf_model():
    return RandomForestRegressor(

        # =========================
        #  核心模型参数（重点调参区）
        # =========================
        n_estimators=300,          # 【参数4】树的数量（100~1000）
        max_depth=None,            # 【参数5】树深度（None=无限 / 10~50）
        min_samples_split=2,       # 【参数6】节点最小分裂样本数
        min_samples_leaf=1,        # 【参数7】叶子节点最小样本数
        max_features='sqrt',       # 【参数8】特征采样（'sqrt','log2',None）

        # =========================
        # Bagging相关
        # =========================
        bootstrap=True,            # 固定（建议True）
        max_samples=None,          # 可调（0.7~1.0）

        # =========================
        # 并行与随机性
        # =========================
        n_jobs=-1,                 # 固定：全核并行
        random_state=42            # 【参数9】随机种子（影响结果）

    )


# =========================================================
# 交叉验证训练
# =========================================================
for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
    print(f"\n=== Fold {fold_idx + 1}/5 ===")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # =========================
    # 训练 RF
    # =========================
    model = build_rf_model()
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)
    all_preds[test_idx] = y_pred

    # =========================
    # 评估
    # =========================
    fold_mae = mean_absolute_error(y_test, y_pred)
    fold_r2 = r2_score(y_test, y_pred)
    fold_rmse = rmse(y_test, y_pred)

    fold_results.append({
        "Fold": fold_idx + 1,
        "MAE": fold_mae,
        "R2": fold_r2,
        "RMSE": fold_rmse
    })

    print(f"MAE: {fold_mae:.4f} | R2: {fold_r2:.4f} | RMSE: {fold_rmse:.4f}")


# =========================================================
# 平均结果
# =========================================================
print("\n=== 交叉验证平均结果 ===")
print(f"MAE : {np.mean([f['MAE'] for f in fold_results]):.4f}")
print(f"R2  : {np.mean([f['R2'] for f in fold_results]):.4f}")
print(f"RMSE: {np.mean([f['RMSE'] for f in fold_results]):.4f}")


# =========================================================
# 全数据评估
# =========================================================
print("\n=== 完整数据集评估 ===")
mae = mean_absolute_error(y, all_preds)
mse = mean_squared_error(y, all_preds)
r2 = r2_score(y, all_preds)

print("MAE  :", mae)
print("MSE  :", mse)
print("R2   :", r2)
print("RMSE :", rmse(y, all_preds))
print("NRMSE:", nrmse(y, all_preds))
print("RMSLE:", rmsle(y, all_preds))


# =========================================================
# 各目标变量评估
# =========================================================
print("\n=== 各目标变量评估 ===")
for i, col in enumerate(target_columns):
    mae_i = mean_absolute_error(y[:, i], all_preds[:, i])
    r2_i = r2_score(y[:, i], all_preds[:, i])

    print(f"{col} - MAE:{mae_i:.3f} | R2:{r2_i:.3f} | "
          f"RMSE:{rmse(y[:, i], all_preds[:, i]):.3f} | "
          f"NRMSE:{nrmse(y[:, i], all_preds[:, i]):.3f} | "
          f"RMSLE:{rmsle(y[:, i], all_preds[:, i]):.3f}")


# =========================================================
# 可视化
# =========================================================
plt.figure(figsize=(15, 10))

for i, col in enumerate(target_columns):
    plt.subplot(3, 2, i + 1)
    plt.plot(y[:, i], label='Actual', marker='o', markersize=2)
    plt.plot(all_preds[:, i], label='Predicted', marker='x', markersize=2)

    plt.title(f'Random Forest Prediction - {col}')
    plt.xlabel('Sample Index')
    plt.ylabel(col)
    plt.legend()
    plt.grid(True)

plt.tight_layout()




# =========================================================
# 导出结果
# =========================================================
comparison_df = pd.DataFrame()

for i, col in enumerate(target_columns):
    comparison_df[f"{col}_Actual"] = y[:, i]
    comparison_df[f"{col}_Predicted"] = all_preds[:, i]

# 加入特征（可选）
for j, col in enumerate(feature_columns):
    comparison_df[col] = X[:, j]

# 输出文件名
comparison_df.to_csv("random_forest_total_prediction.csv", index=False)

print("\n已保存结果：random_forest_total_prediction.csv")