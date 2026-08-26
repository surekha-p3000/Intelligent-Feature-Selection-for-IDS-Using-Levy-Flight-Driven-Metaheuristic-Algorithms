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

# Optimization algorithm parameters (used for both CFO and SSO)
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
# Fitness Function (compatible with continuous optimization output)
# ----------------------------
def fitness_function(features, X, y, w1, w2, w3):
    """Train RandomForestClassifier on selected features and return fitness based on formula."""
    # Select features where the continuous value is > 0.5
    selected_features = np.where(features > 0.5)[0]
    if len(selected_features) == 0:
        return 0 # Return 0 fitness if no features are selected

    X_selected = X.iloc[:, selected_features]

    if X_selected.shape[0] != y.shape[0]:
        print("Warning: Mismatch in samples between X_selected and y")
        return 0

    # Estimate computational cost (simple approach: time to train the model)
    start_time = time.time()
    # Use a smaller n_estimators for faster fitness evaluation during optimization
    clf = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)
    try:
        clf.fit(X_selected, y)
        end_time = time.time()
        comp_cost = end_time - start_time

        # Calculate metrics on the *training* data for the fitness function
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
        # Calculate incorrect prediction rate as a proxy for False Positives in multi-class
        false_positive_proxy = incorrect_predictions / total_predictions if total_predictions > 0 else 0

        # Calculate fitness using the formula
        fitness = w1 * accuracy + w2 * (1 - false_positive_proxy) - w3 * comp_cost

        return fitness

    except ValueError as e:
        print(f"Error during RandomForestClassifier fit in fitness function: {e}")
        return 0
    except Exception as e:
        print(f"An unexpected error occurred in fitness function: {e}")
        return 0

# ----------------------------
# Social Spider Optimization Algorithm (SSO)
# ----------------------------
def social_spider_optimization(X, y, dim, pop_size, iterations, lb, ub, w1, w2, w3):
    """
    Social Spider Optimization algorithm for feature selection.
    """
    # Step 1: Initialization
    population = lb + (ub - lb) * np.random.rand(pop_size, dim)
    fitness = np.zeros(pop_size)
    gender = np.zeros(pop_size, dtype=int) # 0 for Female, 1 for Male
    weight = np.zeros(pop_size)

    best_solution = np.zeros(dim)
    best_fitness = -np.inf

    best_fitness_history = []

    for it in range(iterations):
        # Step 2: Fitness Evaluation
        for i in range(pop_size):
            fitness[i] = fitness_function(population[i], X, y, w1, w2, w3)

        # Update global best
        current_best_idx = np.argmax(fitness)
        if fitness[current_best_idx] > best_fitness:
            best_fitness = fitness[current_best_idx]
            best_solution = population[current_best_idx].copy()

        best_fitness_history.append(best_fitness)

        # Step 3: Gender Assignment (Example: 70% Female, 30% Male based on fitness rank)
        sorted_indices = np.argsort(fitness)[::-1] # Sort in descending order of fitness
        num_females = int(pop_size * 0.7)
        gender[sorted_indices[:num_females]] = 0 # Top 70% are female
        gender[sorted_indices[num_females:]] = 1 # Remaining 30% are male

        # Step 4: Weight Calculation
        max_fitness = np.max(fitness)
        min_fitness = np.min(fitness)
        if max_fitness == min_fitness: # Avoid division by zero
             weight = np.ones(pop_size)
        else:
            weight = (fitness - min_fitness) / (max_fitness - min_fitness)

        # Separate females and males
        female_indices = np.where(gender == 0)[0]
        male_indices = np.where(gender == 1)[0]

        best_female_idx = female_indices[np.argmax(fitness[female_indices])] if female_indices.size > 0 else None
        best_male_idx = male_indices[np.argmax(fitness[male_indices])] if male_indices.size > 0 else None

        # Step 5: Social Interaction (Movement)
        new_population = population.copy()

        # Female movement
        for i in female_indices:
            # Movement towards best female
            F_attract = np.random.rand() * (best_solution - population[i]) # Attract towards global best
            if best_female_idx is not None:
                 F_attract += np.random.rand() * (population[best_female_idx] - population[i]) # Attract towards best female

            # Movement towards a random spider j with higher weight (attraction or repulsion)
            j = np.random.choice(pop_size)
            # Ensure j is not the same as i and j has higher weight, if possible
            possible_js = np.where(weight > weight[i])[0]
            if possible_js.size > 0:
                j = np.random.choice(possible_js)
                F_interact = np.random.rand() * (population[j] - population[i])
            else: # If no spider has higher weight, move away from a random spider
                 j = np.random.choice(pop_size)
                 F_interact = np.random.rand() * (population[i] - population[j]) # Repulsion

            # Random movement
            F_rand = np.random.rand(dim) * (ub - lb) * 0.01 # Small random step

            new_population[i] = population[i] + F_attract + F_interact + F_rand

        # Male movement
        # Males move towards the best female
        if male_indices.size > 0 and best_female_idx is not None:
            for i in male_indices:
                new_population[i] = population[i] + np.random.rand() * (population[best_female_idx] - population[i])

        # Apply bounds
        population = np.clip(new_population, lb, ub)

        print(f"Iteration {it+1}/{iterations} - Best Fitness: {best_fitness:.4f}")

    return best_solution, best_fitness, best_fitness_history

# ----------------------------
# Run Social Spider Optimization Algorithm (SSO) on Training Data
# ----------------------------
print("\n🚀 Running Social Spider Optimization...")

# Define bounds for the features (0 to 1 for feature selection)
# SSO will now select from the original features after preprocessing.
dim = X_train.shape[1] # Use the dimension of the original training data
lb = np.zeros(dim)
ub = np.ones(dim)

start_time_sso = time.time()
# Comment out the call to crayfish_optimization
# best_features, best_score, cfo_convergence = crayfish_optimization(
#     X_train, y_train, dim, # Use X_train instead of X_train_reduced
#     population_size, num_iterations, lb, ub,
#     w1, w2, w3 # Pass weights to CFO
# )

# Call the new social_spider_optimization function
best_features, best_score, optimization_convergence = social_spider_optimization(
    X_train, y_train, dim,
    population_size, num_iterations, lb, ub,
    w1, w2, w3
)
end_time_sso = time.time()
optimization_time = end_time_sso - start_time_sso

print("\n✅ Best SSO training fitness:", best_score)
print(f"SSO Optimization Time: {optimization_time:.2f} seconds")

# ----------------------------
# Final Evaluation (using selected original features with different classifiers)
# ----------------------------
selected_indices = np.where(best_features > 0.5)[0]

# Print selected features
selected_feature_names = X_train.columns[selected_indices].tolist()
print(f"\nSelected Features by SSO: {selected_feature_names}")

# Ensure selected_indices is not empty before proceeding
if selected_indices.size == 0:
    print("\n⚠️ No features selected by SSO. Cannot perform final evaluation.")
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

    print("\n🎯 Final Test Metrics (with SSO Feature Selection):")

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


