import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv('intern_course_ratings.csv', index_col=0)
print("Intern-Course rating matrix (0 = not taken):")
print(df)

matrix = df.values  # rows = interns, columns = courses
interns = df.index.tolist()
courses = df.columns.tolist()

# ============================================================
# 2. HOLD OUT SOME KNOWN RATINGS FOR EVALUATION
# ============================================================
# We hide a few real ratings, predict them, and check how close we get.
# This mimics "does the model recommend what the intern actually liked".
np.random.seed(42)
known_positions = np.argwhere(matrix > 0)  # positions where a real rating exists
n_test = 8
test_indices = known_positions[np.random.choice(len(known_positions), n_test, replace=False)]

train_matrix = matrix.copy()
true_values = []
for (i, j) in test_indices:
    true_values.append(train_matrix[i, j])
    train_matrix[i, j] = 0  # hide it from training

# ============================================================
# 3. MATRIX FACTORIZATION (Collaborative Filtering via TruncatedSVD)
# ============================================================
# TruncatedSVD breaks the big sparse matrix into smaller "latent factor"
# matrices - it finds hidden patterns (e.g., "intern likes technical courses")
# without us ever labeling those patterns ourselves.
n_factors = 3  # number of hidden "taste" dimensions to learn
svd = TruncatedSVD(n_components=n_factors, random_state=42)

intern_factors = svd.fit_transform(train_matrix)      # interns -> latent factors
course_factors = svd.components_                       # latent factors -> courses

# Reconstruct the full matrix: this fills in predicted scores for EVERY cell,
# including courses an intern hasn't taken yet.
predicted_matrix = np.dot(intern_factors, course_factors)

# ============================================================
# 4. EVALUATE ON THE HELD-OUT RATINGS
# ============================================================
predicted_values = [predicted_matrix[i, j] for (i, j) in test_indices]
rmse = mean_squared_error(true_values, predicted_values) ** 0.5

print(f"\nHeld-out actual ratings:    {[round(v, 1) for v in true_values]}")
print(f"Held-out predicted ratings: {[round(v, 1) for v in predicted_values]}")
print(f"RMSE on held-out ratings: {rmse:.2f}")

# ============================================================
# 5. GENERATE TOP RECOMMENDATIONS FOR EACH INTERN
# ============================================================
def recommend_for_intern(intern_name, top_n=3):
    idx = interns.index(intern_name)
    already_taken = matrix[idx] > 0
    scores = predicted_matrix[idx].copy()
    scores[already_taken] = -np.inf  # don't recommend what they've already done
    top_course_idx = np.argsort(scores)[::-1][:top_n]
    return [(courses[i], round(predicted_matrix[idx, i], 2)) for i in top_course_idx]

print("\nSample recommendations:")
for name in ['Intern_1', 'Intern_5', 'Intern_11']:
    recs = recommend_for_intern(name)
    print(f"\n{name} (already taken: {[c for c, v in zip(courses, matrix[interns.index(name)]) if v > 0]})")
    for course, score in recs:
        print(f"  -> Recommend: {course} (predicted score: {score})")
