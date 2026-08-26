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



def ant_lion_optimization(X, y, dim, num_ants, iterations, lb, ub, w1, w2, w3):
    """
    Ant Lion Optimization Algorithm for feature selection.
    """
    # Ensure num_ants and num_antlions are consistent (usually equal or num_antlions < num_ants)
    num_antlions = num_ants # For simplicity, let's use the same number

    # Step 2: Initialization
    antlions = lb + (ub - lb) * np.random.rand(num_antlions, dim)
    ants = lb + (ub - lb) * np.random.rand(num_ants, dim)

    # Step 3: Calculate initial fitness
    antlions_fitness = np.array([fitness_function(antlion, X, y, w1, w2, w3) for antlion in antlions])
    ants_fitness = np.array([fitness_function(ant, X, y, w1, w2, w3) for ant in ants])

    # Step 4: Identify the initial best Antlion
    best_antlion_idx = np.argmax(antlions_fitness)
    best_antlion = antlions[best_antlion_idx].copy()
    best_antlion_fitness = antlions_fitness[best_antlion_idx]

    # Store best fitness at each iteration
    best_fitness_history = [best_antlion_fitness]

    # Step 5: Main optimization loop
    for it in range(iterations):
        # Update bounds for random walks (shrink with iterations)
        # This is a common adaptation in ALO to balance exploration and exploitation
        I = 1 + it / iterations
        lower_bound_walk = lb / I
        upper_bound_walk = ub / I

        for i in range(num_ants):
            # Step 6a: Select an Antlion using roulette wheel selection
            # Ensure fitness values are non-negative for roulette wheel
            positive_antlions_fitness = antlions_fitness - np.min(antlions_fitness) + 1e-6 # Add small value to avoid zero
            selection_probs = positive_antlions_fitness / np.sum(positive_antlions_fitness)
            selected_antlion_idx = np.random.choice(num_antlions, p=selection_probs)
            selected_antlion = antlions[selected_antlion_idx]

            # Simulate random walk of the Ant around the selected Antlion
            # Step 6b & 6c: Simulate random walk and update Ant's position
            # Simple random walk (can be replaced with Levy flight)
            # Ensure the walk is relative to the antlion's position and within bounds
            random_walk_antlion = selected_antlion + (np.random.rand(dim) - 0.5) * (upper_bound_walk - lower_bound_walk)
            random_walk_ant = ants[i] + (np.random.rand(dim) - 0.5) * (upper_bound_walk - lower_bound_walk)

            # Combine walks (simplified ALO step)
            # A more detailed ALO would simulate walks around the antlion's position
            # and the position of the current best antlion
            new_ant_position = (random_walk_antlion + random_walk_ant) / 2

            # Apply bounds
            new_ant_position = np.clip(new_ant_position, lb, ub)

            # Step 6d: Calculate fitness of the updated Ant position
            new_ant_fitness = fitness_function(new_ant_position, X, y, w1, w2, w3)

            # Step 6e: If the Ant's new fitness is better, update its position and fitness
            if new_ant_fitness > ants_fitness[i]:
                ants[i] = new_ant_position
                ants_fitness[i] = new_ant_fitness

        # Step 7: Update Antlion positions
        # If an Ant's fitness is better than an Antlion's, the Ant replaces the Antlion
        for i in range(num_ants):
            for j in range(num_antlions):
                if ants_fitness[i] > antlions_fitness[j]:
                    antlions[j] = ants[i].copy()
                    antlions_fitness[j] = ants_fitness[i]
                    # Reset the ant after it replaces an antlion
                    # ants[i] = lb + (ub - lb) * np.random.rand(dim) # Optional: Reinitialize ant
                    # ants_fitness[i] = fitness_function(ants[i], X, y, w1, w2, w3) # Optional: Recalculate fitness

        # Step 8: Update the best Antlion
        current_best_antlion_idx = np.argmax(antlions_fitness)
        current_best_antlion_fitness = antlions_fitness[current_best_antlion_idx]

        if current_best_antlion_fitness > best_antlion_fitness:
            best_antlion = antlions[current_best_antlion_idx].copy()
            best_antlion_fitness = current_best_antlion_fitness

        # Step 9: Store the best fitness of each iteration
        best_fitness_history.append(best_antlion_fitness)

        print(f"Iteration {it+1}/{iterations} - Best Antlion Fitness: {best_antlion_fitness:.4f}")

    # Step 10: Return the best Antlion's position, its fitness, and the history
    return best_antlion, best_antlion_fitness, best_fitness_history

# ----------------------------
# Run Ant Lion Optimization Algorithm (ALO) on Training Data
# ----------------------------
print("\n🚀 Running Ant Lion Optimization (ALO)...")

# Define bounds for the features (0 to 1 for feature selection)
dim = X_train.shape[1] # Use the dimension of the original training data
lb = np.zeros(dim)
ub = np.ones(dim)

# ALO parameters (can be tuned)
num_ants = population_size # Use the same population size as CFO for comparison
alo_iterations = num_iterations # Use the same number of iterations as CFO

start_time_alo = time.time()
best_features_alo, best_score_alo, alo_convergence = ant_lion_optimization(
    X_train, y_train, dim, # Use X_train
    num_ants, alo_iterations, lb, ub,
    w1, w2, w3 # Pass weights to ALO
)
end_time_alo = time.time()
alo_time = end_time_alo - start_time_alo

print("\n✅ Best ALO training fitness:", best_score_alo)
print(f"ALO Optimization Time: {alo_time:.2f} seconds")

# ----------------------------
# Final Evaluation (using selected original features with different classifiers)
# ----------------------------
selected_indices_alo = np.where(best_features_alo > 0.5)[0]

# Print selected features
selected_feature_names_alo = X_train.columns[selected_indices_alo].tolist()
print(f"\nSelected Features by ALO: {selected_feature_names_alo}")

# Ensure selected_indices_alo is not empty before proceeding
if selected_indices_alo.size == 0:
    print("\n⚠️ No features selected by ALO. Cannot perform final evaluation.")
else:
    # Correct indexing for pandas DataFrame
    X_train_selected_alo = X_train.iloc[:, selected_indices_alo]
    X_test_selected_alo = X_test.iloc[:, selected_indices_alo]

    # Initialize classifiers (same as before)
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear'),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "Bagging": BaggingClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        "SVM": SVC(random_state=42),
        "LightGBM": LGBMClassifier(random_state=42),
        "Gaussian Naive Bayes": GaussianNB()
    }

    print("\n🎯 Final Test Metrics (with ALO Feature Selection):")

    for name, clf in classifiers.items():
        print(f"\n--- {name} ---")
        start_time_eval = time.time()
        try:
            # Train and predict using the features selected by ALO
            clf.fit(X_train_selected_alo, y_train)
            y_pred = clf.predict(X_test_selected_alo)
            end_time_eval = time.time()
            eval_time = end_time_eval - start_time_eval

            print("Accuracy:", accuracy_score(y_test, y_pred))
            print("Precision:", precision_score(y_test, y_pred, average='weighted', zero_division=0))
            print("Recall:", recall_score(y_test, y_pred, average='weighted', zero_division=0))
            print("F1 Score:", f1_score(y_test, y_pred, average='weighted', zero_division=0))
            print(f"Model Training and Prediction Time: {eval_time:.2f} seconds")
        except Exception as e:
            print(f"Error training or predicting with {name}: {e}")
            print("Skipping evaluation for this classifier.")


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
# Run Ant Lion Optimization Algorithm (ALO) on Training Data
# ----------------------------
print("\n🚀 Running Ant Lion Optimization (ALO)...")

# Define bounds for the features (0 to 1 for feature selection)
dim = X_train.shape[1] # Use the dimension of the original training data
lb = np.zeros(dim)
ub = np.ones(dim)

# ALO parameters (can be tuned)
num_ants = population_size # Use the same population size as CFO for comparison
alo_iterations = num_iterations # Use the same number of iterations as CFO

start_time_alo = time.time()
best_features_alo, best_score_alo, alo_convergence = ant_lion_optimization(
    X_train, y_train, dim, # Use X_train
    num_ants, alo_iterations, lb, ub,
    w1, w2, w3 # Pass weights to ALO
)
end_time_alo = time.time()
alo_time = end_time_alo - start_time_alo

print("\n✅ Best ALO training fitness:", best_score_alo)
print(f"ALO Optimization Time: {alo_time:.2f} seconds")

# ----------------------------
# Final Evaluation (using selected original features with different classifiers)
# ----------------------------
selected_indices_alo = np.where(best_features_alo > 0.5)[0]

# Print selected features
selected_feature_names_alo = X_train.columns[selected_indices_alo].tolist()
print(f"\nSelected Features by ALO: {selected_feature_names_alo}")

# Ensure selected_indices_alo is not empty before proceeding
if selected_indices_alo.size == 0:
    print("\n⚠️ No features selected by ALO. Cannot perform final evaluation.")
else:
    # Correct indexing for pandas DataFrame
    X_train_selected_alo = X_train.iloc[:, selected_indices_alo]
    X_test_selected_alo = X_test.iloc[:, selected_indices_alo]

    # Initialize classifiers (same as before)
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear'),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "Bagging": BaggingClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        "SVM": SVC(random_state=42),
        "LightGBM": LGBMClassifier(random_state=42),
        "Gaussian Naive Bayes": GaussianNB()
    }

    print("\n🎯 Final Test Metrics (with ALO Feature Selection):")

    for name, clf in classifiers.items():
        print(f"\n--- {name} ---")
        start_time_eval = time.time()
        try:
            # Train and predict using the features selected by ALO
            clf.fit(X_train_selected_alo, y_train)
            y_pred = clf.predict(X_test_selected_alo)
            end_time_eval = time.time()
            eval_time = end_time_eval - start_time_eval

            print("Accuracy:", accuracy_score(y_test, y_pred))
            print("Precision:", precision_score(y_test, y_pred, average='weighted', zero_division=0))
            print("Recall:", recall_score(y_test, y_pred, average='weighted', zero_division=0))
            print("F1 Score:", f1_score(y_test, y_pred, average='weighted', zero_division=0))
            print(f"Model Training and Prediction Time: {eval_time:.2f} seconds")
        except Exception as e:
            print(f"Error training or predicting with {name}: {e}")
            print("Skipping evaluation for this classifier.")
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

# ALO parameters
population_size = 20 # Used for num_ants and num_antlions
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
# Run Ant Lion Optimization Algorithm (ALO) on Training Data
# ----------------------------
print("\n🚀 Running Ant Lion Optimization (ALO)...")

# Define bounds for the features (0 to 1 for feature selection)
dim = X_train.shape[1] # Use the dimension of the original training data
lb = np.zeros(dim)
ub = np.ones(dim)

# ALO parameters (can be tuned)
num_ants = population_size # Use the same population size as CFO for comparison
alo_iterations = num_iterations # Use the same number of iterations as CFO

start_time_alo = time.time()
best_features_alo, best_score_alo, alo_convergence = ant_lion_optimization(
    X_train, y_train, dim, # Use X_train
    num_ants, alo_iterations, lb, ub,
    w1, w2, w3 # Pass weights to ALO
)
end_time_alo = time.time()
alo_time = end_time_alo - start_time_alo

print("\n✅ Best ALO training fitness:", best_score_alo)
print(f"ALO Optimization Time: {alo_time:.2f} seconds")

# ----------------------------
# Final Evaluation (using selected original features with different classifiers)
# ----------------------------
selected_indices_alo = np.where(best_features_alo > 0.5)[0]

# Print selected features
selected_feature_names_alo = X_train.columns[selected_indices_alo].tolist()
print(f"\nSelected Features by ALO: {selected_feature_names_alo}")

# Ensure selected_indices_alo is not empty before proceeding
if selected_indices_alo.size == 0:
    print("\n⚠️ No features selected by ALO. Cannot perform final evaluation.")
else:
    # Correct indexing for pandas DataFrame
    X_train_selected_alo = X_train.iloc[:, selected_indices_alo]
    X_test_selected_alo = X_test.iloc[:, selected_indices_alo]

    # Initialize classifiers (same as before)
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear'),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "Bagging": BaggingClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        "SVM": SVC(random_state=42),
        "LightGBM": LGBMClassifier(random_state=42),
        "Gaussian Naive Bayes": GaussianNB()
    }

    print("\n🎯 Final Test Metrics (with ALO Feature Selection):")

    for name, clf in classifiers.items():
        print(f"\n--- {name} ---")
        start_time_eval = time.time()
        try:
            # Train and predict using the features selected by ALO
            clf.fit(X_train_selected_alo, y_train)
            y_pred = clf.predict(X_test_selected_alo)
            end_time_eval = time.time()
            eval_time = end_time_eval - start_time_eval

            print("Accuracy:", accuracy_score(y_test, y_pred))
            print("Precision:", precision_score(y_test, y_pred, average='weighted', zero_division=0))
            print("Recall:", recall_score(y_test, y_pred, average='weighted', zero_division=0))
            print("F1 Score:", f1_score(y_test, y_pred, average='weighted', zero_division=0))
            print(f"Model Training and Prediction Time: {eval_time:.2f} seconds")
        except Exception as e:
            print(f"Error training or predicting with {name}: {e}")
            print("Skipping evaluation for this classifier.")

 
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

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

# ALO parameters
population_size = 20 # Used for num_ants and num_antlions
num_iterations = 30

# Fitness function weights
w1 = 1.0  # Weight for Accuracy
w2 = 1.0  # Weight for (1 - False Positives)
w3 = 0.1  # Weight for Computational cost (adjust as needed)

# Define fitness_function (already defined in previous successful cell, but included for completeness)
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

# Define ant_lion_optimization function (already defined in previous successful cell, but included for completeness)
def ant_lion_optimization(X, y, dim, num_ants, iterations, lb, ub, w1, w2, w3):
    """
    Ant Lion Optimization Algorithm for feature selection.
    """
    # Ensure num_ants and num_antlions are consistent (usually equal or num_antlions < num_ants)
    num_antlions = num_ants # For simplicity, let's use the same number

    # Step 2: Initialization
    antlions = lb + (ub - lb) * np.random.rand(num_antlions, dim)
    ants = lb + (ub - lb) * np.random.rand(num_ants, dim)

    # Step 3: Calculate initial fitness
    antlions_fitness = np.array([fitness_function(antlion, X, y, w1, w2, w3) for antlion in antlions])
    ants_fitness = np.array([fitness_function(ant, X, y, w1, w2, w3) for ant in ants])

    # Step 4: Identify the initial best Antlion
    best_antlion_idx = np.argmax(antlions_fitness)
    best_antlion = antlions[best_antlion_idx].copy()
    best_antlion_fitness = antlions_fitness[best_antlion_idx]

    # Store best fitness at each iteration
    best_fitness_history = [best_antlion_fitness]

    # Step 5: Main optimization loop
    for it in range(iterations):
        # Update bounds for random walks (shrink with iterations)
        # This is a common adaptation in ALO to balance exploration and exploitation
        I = 1 + it / iterations
        lower_bound_walk = lb / I
        upper_bound_walk = ub / I

        for i in range(num_ants):
            # Step 6a: Select an Antlion using roulette wheel selection
            # Ensure fitness values are non-negative for roulette wheel
            positive_antlions_fitness = antlions_fitness - np.min(antlions_fitness) + 1e-6 # Add small value to avoid zero
            selection_probs = positive_antlions_fitness / np.sum(positive_antlions_fitness)
            selected_antlion_idx = np.random.choice(num_antlions, p=selection_probs)
            selected_antlion = antlions[selected_antlion_idx]

            # Simulate random walk of the Ant around the selected Antlion
            # Step 6b & 6c: Simulate random walk and update Ant's position
            # Simple random walk (can be replaced with Levy flight)
            # Ensure the walk is relative to the antlion's position and within bounds
            random_walk_antlion = selected_antlion + (np.random.rand(dim) - 0.5) * (upper_bound_walk - lower_bound_walk)
            random_walk_ant = ants[i] + (np.random.rand(dim) - 0.5) * (upper_bound_walk - lower_bound_walk)

            # Combine walks (simplified ALO step)
            # A more detailed ALO would simulate walks around the antlion's position
            # and the position of the current best antlion
            new_ant_position = (random_walk_antlion + random_walk_ant) / 2

            # Apply bounds
            new_ant_position = np.clip(new_ant_position, lb, ub)

            # Step 6d: Calculate fitness of the updated Ant position
            new_ant_fitness = fitness_function(new_ant_position, X, y, w1, w2, w3)

            # Step 6e: If the Ant's new fitness is better, update its position and fitness
            if new_ant_fitness > ants_fitness[i]:
                ants[i] = new_ant_position
                ants_fitness[i] = new_ant_fitness

        # Step 7: Update Antlion positions
        # If an Ant's fitness is better than an Antlion's, the Ant replaces the Antlion
        for i in range(num_ants):
            for j in range(num_antlions):
                if ants_fitness[i] > antlions_fitness[j]:
                    antlions[j] = ants[i].copy()
                    antlions_fitness[j] = ants_fitness[i]
                    # Reset the ant after it replaces an antlion
                    # ants[i] = lb + (ub - lb) * np.random.rand(dim) # Optional: Reinitialize ant
                    # ants_fitness[i] = fitness_function(ants[i], X, y, w1, w2, w3) # Optional: Recalculate fitness

        # Step 8: Update the best Antlion
        current_best_antlion_idx = np.argmax(antlions_fitness)
        current_best_antlion_fitness = antlions_fitness[current_best_antlion_idx]

        if current_best_antlion_fitness > best_antlion_fitness:
            best_antlion = antlions[current_best_antlion_idx].copy()
            best_antlion_fitness = current_best_antlion_fitness

        # Step 9: Store the best fitness of each iteration
        best_fitness_history.append(best_antlion_fitness)

        print(f"Iteration {it+1}/{iterations} - Best Antlion Fitness: {best_antlion_fitness:.4f}")

    # Step 10: Return the best Antlion's position, its fitness, and the history
    return best_antlion, best_antlion_fitness, best_fitness_history

# ----------------------------
# Load and Concatenate (Re-executing this section)
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
# Stratified Downsampling with Cap (Re-executing this section)
# ----------------------------
balanced_df = (
    df.groupby(label_column, group_keys=False)
      .apply(lambda x: x.sample(min(len(x), max_per_class), random_state=42))
)

print("\n✅ Stratified and capped downsampling applied")
print("Final class distribution:")
print(balanced_df[label_column].value_counts())

# ----------------------------
# Prepare Features + Labels (Re-executing this section)
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

# Train-Test Split (Re-executing this section)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=test_size, stratify=y_encoded, random_state=42
)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# ----------------------------
# Run Ant Lion Optimization Algorithm (ALO) on Training Data
# ----------------------------
print("\n🚀 Running Ant Lion Optimization (ALO)...")

# Define bounds for the features (0 to 1 for feature selection)
dim = X_train.shape[1] # Use the dimension of the original training data
lb = np.zeros(dim)
ub = np.ones(dim)

# ALO parameters (can be tuned)
num_ants = population_size # Use the same population size as CFO for comparison
alo_iterations = num_iterations # Use the same number of iterations as CFO

start_time_alo = time.time()
best_features_alo, best_score_alo, alo_convergence = ant_lion_optimization(
    X_train, y_train, dim, # Use X_train
    num_ants, alo_iterations, lb, ub,
    w1, w2, w3 # Pass weights to ALO
)
end_time_alo = time.time()
alo_time = end_time_alo - start_time_alo

print("\n✅ Best ALO training fitness:", best_score_alo)
print(f"ALO Optimization Time: {alo_time:.2f} seconds")

# ----------------------------
# Final Evaluation (using selected original features with different classifiers)
# ----------------------------
selected_indices_alo = np.where(best_features_alo > 0.5)[0]

# Print selected features
selected_feature_names_alo = X_train.columns[selected_indices_alo].tolist()
print(f"\nSelected Features by ALO: {selected_feature_names_alo}")

# Ensure selected_indices_alo is not empty before proceeding
if selected_indices_alo.size == 0:
    print("\n⚠️ No features selected by ALO. Cannot perform final evaluation.")
else:
    # Correct indexing for pandas DataFrame
    X_train_selected_alo = X_train.iloc[:, selected_indices_alo]
    X_test_selected_alo = X_test.iloc[:, selected_indices_alo]

    # Initialize classifiers (same as before)
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear'),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "Bagging": BaggingClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        "SVM": SVC(random_state=42),
        "LightGBM": LGBMClassifier(random_state=42),
        "Gaussian Naive Bayes": GaussianNB()
    }

    print("\n🎯 Final Test Metrics (with ALO Feature Selection):")

    for name, clf in classifiers.items():
        print(f"\n--- {name} ---")
        start_time_eval = time.time()
        try:
            # Train and predict using the features selected by ALO
            clf.fit(X_train_selected_alo, y_train)
            y_pred = clf.predict(X_test_selected_alo)
            end_time_eval = time.time()
            eval_time = end_time_eval - start_time_eval

            print("Accuracy:", accuracy_score(y_test, y_pred))
            print("Precision:", precision_score(y_test, y_pred, average='weighted', zero_division=0))
            print("Recall:", recall_score(y_test, y_pred, average='weighted', zero_division=0))
            print("F1 Score:", f1_score(y_test, y_pred, average='weighted', zero_division=0))
            print(f"Model Training and Prediction Time: {eval_time:.2f} seconds")
        except Exception as e:
            print(f"Error training or predicting with {name}: {e}")
            print("Skipping evaluation for this classifier.")
# Plot ALO convergence
plt.figure(figsize=(10, 6))
plt.plot(range(1, alo_iterations + 2), alo_convergence) # alo_iterations + 1 because initial fitness is included
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.title("ALO Convergence Plot")
plt.grid(True)
plt.show()

