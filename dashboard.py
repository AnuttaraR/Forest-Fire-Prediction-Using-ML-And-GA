from functools import partial
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_score, \
    recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from time import sleep

# Load cleaned dataset
cleaned_data_path = "cleaned_forestfires.csv"
df = pd.read_csv(cleaned_data_path)
X = df.drop(columns=['log_area'])
y = df['log_area']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Bounds for real-valued genes
LOWER_BOUND = 0.0
UPPER_BOUND = 1.0

# GA Configuration - same as model_with_GA.py
GA_METHODS = {
    "Standard GA": (
        tools.cxTwoPoint,
        tools.mutFlipBit,
        tools.selTournament
    ),
    "Real-Coded GA": (
        partial(tools.cxBlend, alpha=0.5),  # Crossover with alpha
        partial(tools.mutGaussian, mu=0, sigma=1, indpb=0.2),  # Mutation with required parameters
        partial(tools.selTournament, tournsize=3)  # Tournament selection with tournsize=3
    ),
    "NSGA-II": (
        partial(tools.cxSimulatedBinaryBounded, eta=20.0, low=LOWER_BOUND, up=UPPER_BOUND),  # Crossover
        partial(tools.mutPolynomialBounded, eta=20.0, low=LOWER_BOUND, up=UPPER_BOUND, indpb=0.2),  # Mutation
        tools.selNSGA2  # Selection
    ),
    "Hybrid GA": (
        tools.cxOnePoint,
        tools.mutShuffleIndexes,
        partial(tools.selTournament, tournsize=3)
    ),
}

# Feature Names
feature_names = list(X.columns)


# Fitness Function for GA
def evaluate_svm(individual, X_train, y_train, lambda_penalty=200, min_features=3):
    selected_features = [idx for idx, val in enumerate(individual) if val == 1]

    # If no features selected, impose a heavy penalty
    if len(selected_features) == 0:
        return 500.0 + lambda_penalty, 500.0

    # Compute RMSE using the selected features
    X_selected = X_train[:, selected_features]
    model = SVR(kernel='rbf')
    scores = cross_val_score(model, X_selected, y_train, cv=5, scoring='neg_root_mean_squared_error')
    rmse = -scores.mean()

    # Apply penalty if the minimum feature constraint is violated
    if len(selected_features) < min_features:
        rmse += lambda_penalty * (min_features - len(selected_features))

    # Return the penalized RMSE and the number of features
    return rmse, len(selected_features)


# GA Feature Selection
def ga_feature_selection(X, y, _ga_method, n_generations=30):
    n_features = X.shape[1]

    # Initialize the DEAP toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_svm, X_train=X_train, y_train=y_train)
    toolbox.register("mate", _ga_method[0])
    toolbox.register("mutate", _ga_method[1], indpb=0.2)

    # Handle selection method for all GA types
    if _ga_method[2] == tools.selTournament:
        toolbox.register("select", _ga_method[2], tournsize=3)
    else:
        toolbox.register("select", _ga_method[2])

    # Initialize population
    population = toolbox.population(n=50)
    best_fitness, avg_fitness, worst_fitness = [], [], []
    fitness_history = []

    # Evaluate the initial population
    for ind in population:
        del ind.fitness.values  # Ensure no stale fitness values
    fits = map(toolbox.evaluate, population)
    for fit, ind in zip(fits, population):
        ind.fitness.values = fit

    # Start the evolution process
    for gen in range(n_generations):
        # Create offspring
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)

        # Invalidate fitness of offspring
        for ind in offspring:
            del ind.fitness.values

        # Evaluate offspring fitness
        fits = map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        # Combine parents and offspring, and select the next generation
        if _ga_method[2] == tools.selNSGA2:
            population = toolbox.select(population + offspring, k=len(population))  # NSGA-II requires combining
        else:
            population = toolbox.select(offspring, k=len(population))

        # Extract RMSE values (objective 1) for fitness tracking
        rmse_values = [ind.fitness.values[0] for ind in population]
        best_fitness.append(min(rmse_values))
        avg_fitness.append(np.mean(rmse_values))
        worst_fitness.append(max(rmse_values))
        fitness_history.append(rmse_values)

        # Incremental Plot of Population's Fitness Evolution
        fig, ax = plt.subplots()
        for i, gen_fitness in enumerate(fitness_history):
            ax.plot(range(len(gen_fitness)), sorted(gen_fitness), label=f'Generation {i + 1}')
        ax.set_title("Incremental Population Fitness Evolution")
        ax.set_xlabel("Individuals (Sorted)")
        ax.set_ylabel("Fitness (RMSE)")
        ax.legend(loc='upper right')
        st.pyplot(fig)
        sleep(0.5)  # Pause to simulate dynamic update

    # Return the best individual (Pareto-optimal for NSGA-II)
    best_individual = tools.selBest(population, k=1)[0]
    return best_individual, best_fitness, avg_fitness, worst_fitness, fitness_history


# Streamlit UI
def main():
    st.title("GA-based Feature Selection for SVM")

    # Sidebar for GA selection
    ga_option = st.sidebar.selectbox("Select GA Type", list(GA_METHODS.keys()))
    run_ga = st.sidebar.button("Run GA")
    st.sidebar.write("\n\n\n")

    # Baseline SVM
    st.header("Baseline SVM Performance")
    baseline_svm = SVR(kernel='rbf')
    baseline_svm.fit(X_train, y_train)
    baseline_predictions = baseline_svm.predict(X_test)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_predictions))
    baseline_mae = mean_absolute_error(y_test, baseline_predictions)
    baseline_r2 = r2_score(y_test, baseline_predictions)
    st.write(f"Baseline SVM RMSE: {baseline_rmse:.4f}")
    st.write(f"Baseline SVM MAE: {baseline_mae:.4f}")
    st.write(f"Baseline SVM R² Score: {baseline_r2:.4f}")

    # Run GA
    if run_ga:
        st.header("GA Feature Selection")
        crossover, mutation, selection = GA_METHODS[ga_option]
        creator.create("FitnessMin", base.Fitness,
                       weights=(-1.0, -0.5))  # Weights: (-1.0) for RMSE, (-0.5) for features
        creator.create("Individual", list, fitness=creator.FitnessMin)
        best_individual, best_fitness, avg_fitness, worst_fitness, fitness_history = ga_feature_selection(X, y, (
        crossover, mutation, selection))

        # Display Fitness Evolution
        st.subheader("GA Fitness Evolution")
        fig, ax = plt.subplots()
        ax.plot(range(len(best_fitness)), best_fitness, label='Best Fitness')
        ax.plot(range(len(avg_fitness)), avg_fitness, label='Avg Fitness')
        ax.plot(range(len(worst_fitness)), worst_fitness, label='Worst Fitness')
        ax.set_xlabel("Generations")
        ax.set_ylabel("Fitness")
        ax.legend()
        st.pyplot(fig)

        # Display Selected Features
        selected_features = [i for i, val in enumerate(best_individual) if val == 1]
        selected_feature_names = [feature_names[i] for i in selected_features]
        st.write(f"Best RMSE: {best_individual.fitness.values[0]:.4f}")
        st.write(f"Number of Selected Features: {best_individual.fitness.values[1]}")
        st.write("Selected Features:", selected_feature_names)

        # SVM with Selected Features
        X_selected_train = X_train[:, selected_features]
        X_selected_test = X_test[:, selected_features]
        svm_ga = SVR(kernel='rbf')
        svm_ga.fit(X_selected_train, y_train)
        predictions_ga = svm_ga.predict(X_selected_test)
        rmse_ga = np.sqrt(mean_squared_error(y_test, predictions_ga))
        mae_ga = mean_absolute_error(y_test, predictions_ga)
        r2_ga = r2_score(y_test, predictions_ga)
        st.header("SVM with GA-Selected Features")
        st.write(f"SVM RMSE with GA: {rmse_ga:.4f}")
        st.write(f"SVM MAE with GA: {mae_ga:.4f}")
        st.write(f"SVM R² Score with GA: {r2_ga:.4f}")

        # Comparison Table
        st.subheader("Performance Comparison")
        comparison_data = {
            "Metric": ["RMSE", "MAE", "R² Score"],
            "Baseline SVM": [baseline_rmse, baseline_mae, baseline_r2],
            "SVM with GA": [rmse_ga, mae_ga, r2_ga]
        }
        comparison_df = pd.DataFrame(comparison_data)
        st.table(comparison_df)

        # Map Visualization with Actual Burned Area
        st.subheader("Actual vs Predicted Burned Area on Random Map")
        random_coords = np.random.rand(len(y_test), 2) * 100  # Random map coordinates
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter_actual = ax.scatter(random_coords[:, 0], random_coords[:, 1], c=y_test, cmap='Reds',
                                    label='Actual Burned Area', s=50)
        scatter_pred = ax.scatter(random_coords[:, 0], random_coords[:, 1], c=predictions_ga, cmap='Blues', marker='x',
                                  label='Predicted Burned Area', s=50)
        ax.set_title("Comparison of Actual vs Predicted Burned Area")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        cbar = plt.colorbar(scatter_actual, ax=ax, orientation="vertical")
        cbar.set_label("Burned Area (log scale)")
        ax.legend(loc="lower right")
        st.pyplot(fig)

        # Bubble Size Representation
        st.subheader("Burned Area Representation with Bubble Sizes")
        fig, ax = plt.subplots(figsize=(8, 6))
        actual_sizes = np.exp(y_test) * 10  # Bubble sizes scaled
        pred_sizes = np.exp(predictions_ga) * 10  # Bubble sizes scaled
        ax.scatter(random_coords[:, 0], random_coords[:, 1], s=actual_sizes, c='red', alpha=0.5,
                   label='Actual Burned Area')
        ax.scatter(random_coords[:, 0], random_coords[:, 1], s=pred_sizes, c='blue', alpha=0.3,
                   label='Predicted Burned Area')
        ax.set_title("Bubble Size Representation of Burned Areas")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.legend()
        st.pyplot(fig)


if __name__ == "__main__":
    main()

# streamlit run dashboard.py
