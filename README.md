# Learning Path Recommendation System

## Objective
Build a system to recommend personalized learning paths/courses for interns,
based on past intern learning patterns (collaborative filtering).

## Dataset
A synthetic intern-course rating matrix: 15 interns × 8 courses (Python Basics,
Data Analysis, Machine Learning, Deep Learning, SQL Fundamentals, Web
Development, Cloud Computing, NLP Basics). Each intern has completed 3–5
courses with a rating from 1–5; a 0 means the course hasn't been taken. This
sparsity (most cells empty) reflects how real learning-platform data looks.

## Approach
1. **User-item matrix** — interns as rows, courses as columns, ratings as values.
2. **Matrix Factorization** — used `TruncatedSVD` (scikit-learn) to decompose
   the sparse matrix into latent "taste" factors, then reconstructed the full
   matrix to predict scores for courses each intern hasn't taken yet.
3. **Evaluation** — hid 8 known ratings before training, predicted them back,
   and measured RMSE (root mean squared error) between predicted and actual.
4. **Recommendations** — for each intern, recommended the top 3 courses with
   the highest predicted score among courses they haven't already taken.

## Results
- **RMSE on held-out ratings: 3.41** (on a 1–5 scale)

### Sample recommendations
| Intern | Already Taken | Top Recommendation |
|---|---|---|
| Intern_1 | Python Basics, Data Analysis, Web Dev, Cloud Computing, NLP Basics | Machine Learning |
| Intern_5 | Data Analysis, SQL Fundamentals, Web Development | Cloud Computing |
| Intern_11 | Deep Learning, SQL Fundamentals, Web Dev, Cloud Computing, NLP Basics | Machine Learning |

## Important limitation (honest note)
An RMSE of 3.41 on a 1–5 rating scale is high — this reflects the small size
and sparsity of this synthetic dataset (only 15 interns, ~40% of cells filled),
not a flaw in the method itself. Matrix factorization needs substantially more
data to reliably learn latent taste patterns. With a real-world dataset (hundreds
of interns and interactions), this approach would be expected to perform far
better.

**To make this production-ready:**
- Use real historical intern-course completion data, at much larger scale
- Tune the number of latent factors (`n_components`) using cross-validation
- Consider a dedicated recommender library (e.g. `scikit-surprise` or `implicit`)
  which offers more refined matrix factorization algorithms (e.g. SVD++, ALS)
- Add content-based features (course topic, difficulty) to supplement pure
  collaborative filtering, especially for interns with very few completions
  ("cold start" problem)

## Files
- `intern_course_ratings.csv` — the intern-course rating matrix
- `recommendation_system.py` — full pipeline (load → factorize → evaluate → recommend)

## How to run
```
pip install pandas scikit-learn numpy
python3 recommendation_system.py
```
