import pandas as pd
import numpy as np
import time  # Import time for computational cost estimation
from sklearn.model_selection import train_test_split
# from sklearn.decomposition import PCA # Removed PCA import
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score # Import additional metrics
# from sklearn.svm import LinearSVC # Removed LinearSVC import
from sklearn.ensemble import RandomForestClassifier # Added RandomForestClassifier
from sklearn.linear_model import LogisticRegression # Added LogisticRegression
from sklearn.tree import DecisionTreeClassifier # Added DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier # Added KNN
from sklearn.ensemble import BaggingClassifier # Added Bagging
from sklearn.ensemble import AdaBoostClassifier # Added AdaBoost
from xgboost import XGBClassifier # Added XGBoost
from sklearn.svm import SVC # Added SVM
from lightgbm import LGBMClassifier # Added LightGBM
from sklearn.naive_bayes import GaussianNB # Added Gaussian Naive Bayes
import matplotlib.pyplot as plt # Import matplotlib for plotting

# ----------------------------
# Parameters
# ----------------------------
files = [
    "/content/drive/MyDrive/GothamDataset2025/processed/iotsim-city-power-1.csv",
    "/content/drive/MyDrive/GothamDataset2025/processed/iotsim-combined-cycle-1.csv",
    "/content/drive/MyDrive/GothamDataset2025/processed/iotsim-combined-cycle-10.csv"
]

label_column = "label"     # update if your dataset uses a different name
max_per_class = 5000       # cap per class
test_size = 0.2            # 80% train / 20% test

# CFO parameters
population_size = 20
num_iterations = 30

# Fitness function weights
w1 = 1.0  # Weight for Accuracy
w2 = 1.0  # Weight for (1 - False Positives)
w3 = 0.1  # Weight for Computational cost (adjust as needed)

# ----------------------------
# Load and Concatenate
# ----------------------------
df_list = []
for file in files:
    print(f"Loading {file} ...")
    df_single = pd.read_csv(file, low_memory=False)
    print(f"✅ Loaded {file} with {len(df_single)} rows")
    df_list.append(df_single)

df = pd.concat(df_list, ignore_index=True)
print(f"\nAll datasets concatenated successfully. Final dataset shape: {df.shape}")

# ----------------------------
# Stratified Downsampling with Cap
# ----------------------------
balanced_df = (
    df.groupby(label_column, group_keys=False)
      .apply(lambda x: x.sample(min(len(x), max_per_class), random_state=42))
)

print("\n✅ Stratified and capped downsampling applied")
print("Final class distribution:")
print(balanced_df[label_column].value_counts())

# ----------------------------
# Prepare Features + Labels
# ----------------------------
columns_to_drop = [label_column, 'frame.time']
X = balanced_df.drop(columns=columns_to_drop, errors='ignore')
y = balanced_df[label_column]

# Drop columns that are entirely NaN
X = X.dropna(axis=1, how='all')
print(f"Dropped columns with all NaNs. Remaining features: {X.shape[1]}")

# Identify categorical and numerical columns AFTER dropping
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
numerical_cols = X.select_dtypes(include=np.number).columns

# Impute missing values in numerical columns
if numerical_cols.size > 0:
    imputer_num = SimpleImputer(strategy='mean')
    X[numerical_cols] = imputer_num.fit_transform(X[numerical_cols])
    print("Missing values imputed using mean strategy for numerical columns.")
else:
    print("No numerical columns to impute.")

# Impute missing values in categorical columns
if categorical_cols.size > 0:
    imputer_cat = SimpleImputer(strategy='most_frequent')
    X[categorical_cols] = imputer_cat.fit_transform(X[categorical_cols])
    print("Missing values imputed using most frequent strategy for categorical columns.")
else:
    print("No categorical columns to impute.")

# Encode categorical columns
le = LabelEncoder()
for col in categorical_cols:
    X[col] = le.fit_transform(X[col].astype(str))
print("Categorical columns encoded.")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Labels encoded.")

# Drop rows with any remaining NaNs
if X.isnull().sum().sum() > 0:
    print("Warning: Found remaining NaN values in X. Dropping rows with NaNs.")
    X = X.dropna()
    y_encoded = y_encoded[X.index]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=test_size, stratify=y_encoded, random_state=42
)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# ----------------------------
# Removed PCA step
# ----------------------------
# pca = PCA(n_components=0.95)
# X_train_reduced = pca.fit_transform(X_train)
# X_test_reduced = pca.transform(X_test)
# print(f"Original features: {X.shape[1]}, After PCA: {X_train_reduced.shape[1]}")

# ----------------------------
# Crayfish Optimization Algorithm (CFO) - Modified for Random Forest
# ----------------------------
def fitness_function(features, X, y, w1, w2, w3):
    """Train RandomForestClassifier on selected features and return fitness based on formula."""
    selected_features = np.where(features > 0.5)[0]
    if len(selected_features) == 0:
        return 0

    X_selected = X.iloc[:, selected_features]

    if X_selected.shape[0] != y.shape[0]:
        print("Warning: Mismatch in samples between X_selected and y")
        return 0

    # Estimate computational cost (simple approach: time to train the model)
    start_time = time.time()
    clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    try:
        clf.fit(X_selected, y)
        end_time = time.time()
        comp_cost = end_time - start_time

        # Calculate metrics
        y_pred = clf.predict(X_selected)
        accuracy = accuracy_score(y, y_pred)

        # Calculate False Positives (requires confusion matrix)
        # Note: This assumes a binary classification context for 'False Positives'.
        # For multi-class, you might need to define how 'False Positives' are calculated,
        # e.g., micro/macro average, or specify a 'positive' class if applicable.
        # For simplicity here, we'll calculate the proportion of incorrect predictions.
        cm = confusion_matrix(y, y_pred)
        # Sum of off-diagonal elements (incorrect predictions)
        incorrect_predictions = np.sum(cm) - np.trace(cm)
        total_predictions = np.sum(cm)
        false_positive_proxy = incorrect_predictions / total_predictions if total_predictions > 0 else 0

        # Calculate fitness using the formula
        fitness = w1 * accuracy + w2 * (1 - false_positive_proxy) - w3 * comp_cost

        return fitness

    except ValueError as e:
        print(f"Error during RandomForestClassifier fit: {e}")
        return 0

def levy_flight(beta):
    """Generate step using Levy flight."""
    sigma = (
        np.random.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
        (np.random.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))
    ) ** (1 / beta)
    u = np.random.randn() * sigma
    v = np.random.randn()
    step = u / (np.abs(v) ** (1 / beta))
    return step

def crayfish_optimization(X, y, dim, pop_size, iterations, lb, ub, w1, w2, w3):
    """
    Crayfish Optimization Algorithm with temperature stages and Levy Flight.
    Assumes lb and ub are arrays of shape (dim,)
    """
    # Step 1: Initialization
    population = lb + (ub - lb) * np.random.rand(pop_size, dim)
    # Pass DataFrames and weights to fitness function
    fitness = np.array([fitness_function(ind, X, y, w1, w2, w3) for ind in population])

    best_idx = np.argmax(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    # Store best fitness at each iteration
    best_fitness_history = [best_fitness]

    for it in range(iterations):
        # Step 2: Temperature setting
        temp = np.random.rand() * 15 + 20

        for i in range(pop_size):
            new_solution = population[i].copy()

            # Condition 1: If temp > 30 (High temperature - Exploration/Competition)
            if temp > 30:
                # Movement combines random step and potential Levy flight
                r1, r2 = np.random.randint(0, pop_size, 2)
                step = (population[r1] - population[r2]) * np.random.rand()

                # Introduce Levy Flight with a certain probability or always
                # Simple approach: apply Levy flight step scaled by the range
                levy_step = levy_flight(1.5) * (ub - lb) * 0.01 # Beta = 1.5 as example
                new_solution = population[i] + step + levy_step * np.random.randn(dim) # Add randomness to direction

            # Condition 2: If temp <= 30 (Optimal foraging temperature)
            else:
                 # Foraging stage movement (can be similar to original or adapted)
                 # This is a placeholder; actual foraging strategies in CFO are more detailed.
                 # For now, we'll keep a movement similar to the original random step,
                 # potentially refined with Levy Flight for adaptive search.
                r1, r2 = np.random.randint(0, pop_size, 2)
                step = (population[r1] - population[r2]) * np.random.rand()

                # Levy Flight for adaptive search
                levy_step = levy_flight(1.0) * (ub - lb) * 0.005 # Smaller steps for search
                new_solution = population[i] + step + levy_step * np.random.randn(dim)

            # Apply bounds and evaluate
            new_solution = np.clip(new_solution, lb, ub)
            # Pass DataFrames and weights to fitness function
            new_fitness = fitness_function(new_solution, X, y, w1, w2, w3)

            if new_fitness > fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness

                if new_fitness > best_fitness:
                    best_solution = new_solution.copy()
                    best_fitness = new_fitness
        
        best_fitness_history.append(best_fitness)

        print(f"Iteration {it+1}/{iterations} - Best Fitness: {best_fitness:.4f}")

    return best_solution, best_fitness, best_fitness_history

# ----------------------------
# Run CFO on Training Data (using original features)
# ----------------------------
print("\n🚀 Running Crayfish Optimization...")

# Define bounds for the features (0 to 1 for feature selection)
# CFO will now select from the original features after preprocessing.
dim = X_train.shape[1] # Use the dimension of the original training data
lb = np.zeros(dim)
ub = np.ones(dim)

start_time_cfo = time.time()
best_features, best_score, cfo_convergence = crayfish_optimization(
    X_train, y_train, dim, # Use X_train instead of X_train_reduced
    population_size, num_iterations, lb, ub,
    w1, w2, w3 # Pass weights to CFO
)
end_time_cfo = time.time()
cfo_time = end_time_cfo - start_time_cfo

print("\n✅ Best CFO training fitness:", best_score)
print(f"CFO Optimization Time: {cfo_time:.2f} seconds")

# ----------------------------
# Final Evaluation (using selected original features with different classifiers)
# ----------------------------
selected_indices = np.where(best_features > 0.5)[0]

# Print selected features
selected_feature_names = X_train.columns[selected_indices].tolist()
print(f"\nSelected Features by CFO: {selected_feature_names}")

# Ensure selected_indices is not empty before proceeding
if selected_indices.size == 0:
    print("\n⚠️ No features selected by CFO. Cannot perform final evaluation.")
else:
    # Correct indexing for pandas DataFrame
    X_train_selected = X_train.iloc[:, selected_indices]
    X_test_selected = X_test.iloc[:, selected_indices]

    # Initialize classifiers
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear'), # Added Logistic Regression
        "Decision Tree": DecisionTreeClassifier(random_state=42), # Added Decision Tree
        "KNN": KNeighborsClassifier(), # Added KNN
        "Bagging": BaggingClassifier(random_state=42), # Added Bagging
        "AdaBoost": AdaBoostClassifier(random_state=42), # Added AdaBoost
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42), # Added XGBoost
        "SVM": SVC(random_state=42), # Added SVM
        "LightGBM": LGBMClassifier(random_state=42), # Added LightGBM
        "Gaussian Naive Bayes": GaussianNB() # Added Gaussian Naive Bayes
        # Note: "J48" and "Reduce error pruning DT" are not directly available in scikit-learn.
        # DecisionTreeClassifier is used as an equivalent for J48.
        # For pruning, you might need to explore post-pruning techniques or use libraries
        # that specifically implement reduced error pruning if needed.
        # K-Means is a clustering algorithm and is not included in this classification evaluation.
    }

    print("\n🎯 Final Test Metrics (with CFO Feature Selection):")

    for name, clf in classifiers.items():
        print(f"\n--- {name} ---")
        start_time_eval = time.time()
        # Some classifiers (like SVM and KNN) can benefit from feature scaling
        # If you encounter performance issues or errors, consider adding scaling here
        # scaler = StandardScaler()
        # X_train_scaled = scaler.fit_transform(X_train_selected)
        # X_test_scaled = scaler.transform(X_test_selected)
        # clf.fit(X_train_scaled, y_train)
        # y_pred = clf.predict(X_test_scaled)

        try:
            clf.fit(X_train_selected, y_train)
            y_pred = clf.predict(X_test_selected)
            end_time_eval = time.time()
            eval_time = end_time_eval - start_time_eval

            print("Accuracy:", accuracy_score(y_test, y_pred))
            # Calculate and print other metrics (using average='weighted' for multi-class)
            print("Precision:", precision_score(y_test, y_pred, average='weighted', zero_division=0))
            print("Recall:", recall_score(y_test, y_pred, average='weighted', zero_division=0))
            print("F1 Score:", f1_score(y_test, y_pred, average='weighted', zero_division=0))
            print(f"Model Training and Prediction Time: {eval_time:.2f} seconds")
        except Exception as e:
            print(f"Error training or predicting with {name}: {e}")
            print("Skipping evaluation for this classifier.")
# Plot CFO convergence
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_iterations + 2), cfo_convergence) # num_iterations + 1 because initial fitness is included
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.title("CFO Convergence Plot")
plt.grid(True)
plt.show()

