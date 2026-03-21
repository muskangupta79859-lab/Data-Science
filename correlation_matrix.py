import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = {
    "age": [20, 25, 30, 35, 40],
    "salary": [20000, 30000, 40000, 50000, 60000],
    "experience": [1, 3, 5, 7, 9]
}

df = pd.DataFrame(data)

corr = df.corr()

print("Correlation Matrix:")
print(corr)

# Heatmap
sns.heatmap(corr, annot=True)
plt.title("Correlation Matrix Heatmap")
plt.show()
