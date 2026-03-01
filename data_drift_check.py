import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from scipy.stats import ks_2samp

# 1) Training data load
data = load_iris()
X_train = pd.DataFrame(data.data, columns=data.feature_names)

# 2) Simulate new incoming data (thoda shift + noise)
rng = np.random.RandomState(42)
X_new = X_train.copy()
X_new = X_new + rng.normal(loc=0.2, scale=0.3, size=X_new.shape)

# 3) KS Test se feature-wise drift check
def check_drift(train_df, new_df, alpha=0.05):
    results = []
    for col in train_df.columns:
        stat, p_value = ks_2samp(train_df[col], new_df[col])
        drift = "DRIFT" if p_value < alpha else "NO_DRIFT"
        results.append((col, round(stat, 4), round(p_value, 6), drift))
    return pd.DataFrame(results, columns=["Feature", "KS_Stat", "P_Value", "Status"])

report = check_drift(X_train, X_new)
print(report)

# 4) Simple summary
print("\nDrifted features count:",
      (report["Status"] == "DRIFT").sum(), "/", len(report))
