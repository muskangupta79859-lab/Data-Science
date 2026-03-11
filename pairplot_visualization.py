import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load dataset
data = load_iris()

df = pd.DataFrame(data.data, columns=data.feature_names)
df["species"] = data.target

# Pairplot
sns.pairplot(df, hue="species")

plt.show()
