import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# sample dataset
data = {
    "hours_studied": [1,2,3,4,5,6,7],
    "sleep_hours": [7,6,6,5,5,4,4],
    "exam_score": [50,55,65,70,75,85,90]
}

df = pd.DataFrame(data)

# correlation matrix
corr = df.corr()

# heatmap
sns.heatmap(corr, annot=True, cmap="coolwarm")

plt.title("Correlation Heatmap")
plt.show()
