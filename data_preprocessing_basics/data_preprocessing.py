# Data Preprocessing Basics

# sample raw data
ages=[20, 21, 22, None, 23, None]

# Handling missing values
cleaned_ages = [age for age in ages if age is not None]
print("Cleaned Ages:",cleaned_ages)

# Normalization example
scores = [50,60,70,80]
max_score = max(scores)

normalized_scores = [score / max_score for score in scores]
print("Normalized Scores:",normalized_scores)
