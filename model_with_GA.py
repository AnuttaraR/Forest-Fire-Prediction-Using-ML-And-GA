# Model Building with GA-based Feature Selection
# ------------------------------------------------
# This script loads the cleaned dataset and builds two models: SVM and Random Forest.
# Genetic Algorithms (GAs) are applied for feature selection prior to model training.

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from deap import base, creator, tools, algorithms
import random
import matplotlib.pyplot as plt

# Load the cleaned dataset
cleaned_data_path = "cleaned_forestfires.csv"
df = pd.read_csv(cleaned_data_path)

# Separate features and target
X = df.drop(columns=['log_area'])
y = df['log_area']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Dataset Loaded and Split Successfully")


# Define Genetic Algorithm Components
# ------------------------------------------------
# Fitness Function: RMSE of the selected features with a given model

def evaluate_svm(individual):
    # Convert binary chromosome into selected features
    selected_features = [index for index, value in enumerate(individual) if value == 1]
    if len(selected_features) < 3:  # Minimum features constraint
        return 1000,  # Penalize invalid solutions

    X_selected = X_train[:, selected_features]
    model = SVR(kernel='rbf')
    scores = cross_val_score(model, X_selected, y_train, cv=5, scoring='neg_root_mean_squared_error')
    return -scores.mean(),


def evaluate_rf(individual):
    # Convert binary chromosome into selected features
    selected_features = [index for index, value in enumerate(individual) if value == 1]
    if len(selected_features) < 3:  # Minimum features constraint
        return 1000,  # Penalize invalid solutions

    X_selected = X_train[:, selected_features]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    scores = cross_val_score(model, X_selected, y_train, cv=5, scoring='neg_root_mean_squared_error')
    return -scores.mean(),


# Initialize GA setup
def ga_feature_selection(evaluate_function, model_name, crossover, mutation, selection):
    """
    Runs a Genetic Algorithm for feature selection and evaluates the given model.
    """
    n_features = X.shape[1]

    # DEAP setup
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", crossover)
    toolbox.register("mutate", mutation, indpb=0.2)
    toolbox.register("select", selection, tournsize=3)
    toolbox.register("evaluate", evaluate_function)

    # GA Execution
    population = toolbox.population(n=50)
    n_generations = 30

    print(f"Running GA for {model_name}...")
    for generation in range(n_generations):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
        fits = map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))
        print(f"Generation {generation + 1} Complete")

    # Extract the best solution
    best_individual = tools.selBest(population, k=1)[0]
    selected_features = [index for index, value in enumerate(best_individual) if value == 1]
    print(f"Best Selected Features for {model_name}: {selected_features}")
    return selected_features


# GA Methods Setup
standard_ga = (tools.cxTwoPoint, tools.mutFlipBit, tools.selTournament)
real_coded_ga = (tools.cxBlend, tools.mutGaussian, tools.selRoulette)
nsga_ii = (tools.cxSimulatedBinaryBounded, tools.mutPolynomialBounded, tools.selNSGA2)
hybrid_ga = (tools.cxOnePoint, tools.mutShuffleIndexes, tools.selBest)
differential_evolution = (tools.cxESTwoPoint, tools.mutESLogNormal, tools.selBest)


def run_ga_model(evaluate, model_name, ga_method):
    crossover, mutation, selection = ga_method
    return ga_feature_selection(evaluate, model_name, crossover, mutation, selection)


# Run GA for SVM with all methods
svm_results = {}
for ga_name, ga_method in zip([
    "Standard GA", "Real-Coded GA", "NSGA-II", "Hybrid GA", "Differential Evolution"
], [standard_ga, real_coded_ga, nsga_ii, hybrid_ga, differential_evolution]):
    selected_features = run_ga_model(evaluate_svm, "SVM", ga_method)
    X_train_svm = X_train[:, selected_features]
    X_test_svm = X_test[:, selected_features]

    model = SVR(kernel='rbf')
    model.fit(X_train_svm, y_train)
    predictions = model.predict(X_test_svm)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    svm_results[ga_name] = rmse
    print(f"{ga_name} - SVM RMSE: {rmse:.4f}")

# Run GA for Random Forest with all methods
rf_results = {}
for ga_name, ga_method in zip([
    "Standard GA", "Real-Coded GA", "NSGA-II", "Hybrid GA", "Differential Evolution"
], [standard_ga, real_coded_ga, nsga_ii, hybrid_ga, differential_evolution]):
    selected_features = run_ga_model(evaluate_rf, "Random Forest", ga_method)
    X_train_rf = X_train[:, selected_features]
    X_test_rf = X_test[:, selected_features]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_rf, y_train)
    predictions = model.predict(X_test_rf)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    rf_results[ga_name] = rmse
    print(f"{ga_name} - Random Forest RMSE: {rmse:.4f}")

# Visualize Results
# ------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].bar(svm_results.keys(), svm_results.values(), color='blue')
ax[0].set_title('SVM Model Performance')
ax[0].set_ylabel('RMSE')
ax[0].tick_params(axis='x', rotation=45)

ax[1].bar(rf_results.keys(), rf_results.values(), color='green')
ax[1].set_title('Random Forest Model Performance')
ax[1].set_ylabel('RMSE')
ax[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

print("Model Building with All GA Methods Complete")
