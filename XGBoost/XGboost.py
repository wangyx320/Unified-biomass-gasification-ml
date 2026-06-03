import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

from xgboost import XGBRegressor


# =========================================================
# 指标函数
# =========================================================
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def nrmse(y_true, y_pred):
    return rmse(y_true, y_pred) / (np.max(y_true) - np.min(y_true))


# =========================================================
# 数据
# =========================================================
df = pd.read_csv("2026.csv")

target_columns = ['CO', 'H2', 'CO2', 'CH4', 'LHV']
feature_columns = [c for c in df.columns if c not in target_columns]

X_raw = df[feature_columns].values
y = df[target_columns].values


# =========================================================
# KFold
# =========================================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

all_preds = np.zeros_like(y)
fold_results = []


# =========================================================
# 模型参数（兼容旧版XGBoost）
# =========================================================
xgb_params = {
    "n_estimators": 800,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1
}


# =========================================================
# 交叉验证
# =========================================================
for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_raw)):

    print(f"\n========== Fold {fold_idx + 1}/5 ==========")

    X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    y_pred = np.zeros_like(y_test)

    for i, col in enumerate(target_columns):

        # -------------------------
        # validation split（fold内）
        # -------------------------
        X_train_f, X_val, y_train_f, y_val = train_test_split(
            X_train_raw,
            y_train,
            test_size=0.15,
            random_state=42
        )

        # -------------------------
        # 标准化
        # -------------------------
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_f)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test_raw)

        # -------------------------
        # 模型
        # -------------------------
        model = XGBRegressor(**xgb_params)

        # =====================================================
        # ✔ 关键修复：旧版本 XGBoost 只能用 eval_set，不支持 early stopping
        # =====================================================
        model.fit(
            X_train,
            y_train_f[:, i],
            eval_set=[(X_val, y_val[:, i])],
            verbose=False
        )

        # -------------------------
        # 预测
        # -------------------------
        y_pred[:, i] = model.predict(X_test)

    all_preds[test_idx] = y_pred

    # -------------------------
    # fold评估
    # -------------------------
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
# 总结果
# =========================================================
print("\n========== CV RESULT ==========")
print("Mean MAE :", np.mean([f["MAE"] for f in fold_results]))
print("Mean R2  :", np.mean([f["R2"] for f in fold_results]))
print("Mean RMSE:", np.mean([f["RMSE"] for f in fold_results]))


# =========================================================
# per target
# =========================================================
print("\n========== PER TARGET ==========")

for i, col in enumerate(target_columns):

    r2 = r2_score(y[:, i], all_preds[:, i])

    rmse_val = rmse(y[:, i], all_preds[:, i])

    mae_val = mean_absolute_error(
        y[:, i],
        all_preds[:, i]
    )

    nrmse_val = nrmse(
        y[:, i],
        all_preds[:, i]
    )

    print(
        f"{col}: "
        f"R2 = {r2:.4f} | "
        f"RMSE = {rmse_val:.4f} | "
        f"MAE = {mae_val:.4f} | "
        f"NRMSE = {nrmse_val:.4f}"
    )


# =========================================================
# 可视化
# =========================================================
plt.figure(figsize=(15, 10))

for i, col in enumerate(target_columns):
    plt.subplot(3, 2, i + 1)
    plt.plot(y[:, i], label="Actual")
    plt.plot(all_preds[:, i], label="Predicted")
    plt.title(f"XGBoost - {col}")
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.savefig("xgboost_prediction.pdf", dpi=300)
plt.show()


# =========================================================
# 保存
# =========================================================
result_df = pd.DataFrame()

for i, col in enumerate(target_columns):
    result_df[f"{col}_Actual"] = y[:, i]
    result_df[f"{col}_Predicted"] = all_preds[:, i]

for j, col in enumerate(feature_columns):
    result_df[col] = X_raw[:, j]

result_df.to_csv("xgboost_prediction.csv", index=False)

print("\nSaved: xgboost_prediction.csv")