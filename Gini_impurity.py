import numpy as np

# Dataset
X = np.array([2, 4, 6, 8, 10])   # Study Hours
y = np.array([0, 0, 1, 1, 1])    # Pass/Fail

# Function to calculate Gini impurity
def gini(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))
    gini_score = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row for row in group].count(class_val) / size
            score += p * p
        gini_score += (1.0 - score) * (size / n_instances)
    return gini_score

# Function to test a split
def test_split(index, value, X, y):
    left, right = [], []
    for i in range(len(X)):
        if X[i] < value:
            left.append(y[i])
        else:
            right.append(y[i])
    return left, right

# Find the best split
def get_split(X, y):
    class_values = list(set(y))
    best_gini = 999
    best_value = None
    for value in X:
        groups = test_split(0, value, X, y)
        gini_score = gini(groups, class_values)
        print(f"Split at {value}, Gini: {gini_score:.3f}")
        if gini_score < best_gini:
            best_gini = gini_score
            best_value = value
    return best_value, best_gini

# Run
best_split, best_gini = get_split(X, y)
print(f"\nBest Split: Study Hours < {best_split}, Gini = {best_gini:.3f}")
