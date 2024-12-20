from collections import defaultdict
import joblib
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from deap import base, creator, tools, algorithms
import random
import matplotlib.pyplot as plt
from functools import partial
import scipy.stats as stats
from scipy.stats import t

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

# Define fitness for single-objective and multi-objective optimization
if "FitnessMin" not in creator.__dict__:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Single-objective (minimize RMSE)
if "MultiObjectiveFitnessMin" not in creator.__dict__:
    creator.create("MultiObjectiveFitnessMin", base.Fitness,
                   weights=(-1.0, -0.5))  # NSGA-II (min RMSE, min no.of features)

# Define individuals for single-objective and multi-objective GAs
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMin)
if "MultiObjectiveIndividual" not in creator.__dict__:
    creator.create("MultiObjectiveIndividual", list, fitness=creator.MultiObjectiveFitnessMin)


def compute_confidence_interval(data, confidence=0.95):
    data = np.array(data)
    if len(data) < 2:
        return (np.nan, np.nan)
    mean = np.mean(data)
    sem = np.std(data, ddof=1) / np.sqrt(len(data))  # Standard error of the mean
    ci_lower, ci_upper = t.interval(confidence, len(data) - 1, loc=mean, scale=sem)
    return ci_lower, ci_upper


# Fitness Function: RMSE of the selected features with a given model
def evaluate_svm(individual, lambda_penalty=200, min_features=3):
    selected_features = [idx for idx, val in enumerate(individual) if val == 1]

    # If no features selected, impose a penalty
    if len(selected_features) == 0:
        return (500.0 + lambda_penalty,)

    # Compute RMSE using the selected features
    X_selected = X_train[:, selected_features]
    model = SVR(kernel='rbf')
    scores = cross_val_score(model, X_selected, y_train, cv=5, scoring='neg_root_mean_squared_error')
    rmse = -scores.mean()

    # Apply penalty if the minimum feature constraint is violated
    if len(selected_features) < min_features:
        rmse += lambda_penalty * (min_features - len(selected_features))

    # Return the penalized RMSE as a tuple
    return (rmse,)


def evaluate_svm_nsga(individual, lambda_penalty=200, min_features=3):
    # Objective 1: Reduce RMSE
    # Objective 2: Minimize number of features
    selected_features = [idx for idx, val in enumerate(individual) if val == 1]

    # If no features selected, impose a heavy penalty
    if len(selected_features) == 0:
        return (500.0 + lambda_penalty, len(individual))

    # Compute RMSE using the selected features
    X_selected = X_train[:, selected_features]
    model = SVR(kernel='rbf')
    scores = cross_val_score(model, X_selected, y_train, cv=5, scoring='neg_root_mean_squared_error')
    rmse = -scores.mean()

    # Apply penalty if the minimum feature constraint is violated
    if len(selected_features) < min_features:
        rmse += lambda_penalty * (min_features - len(selected_features))

    # Return the penalized RMSE and the number of selected features
    return (rmse, len(selected_features))


# Genetic Algorithm Methods
LOWER_BOUND = 0.0
UPPER_BOUND = 1.0
GA_METHODS = {
    "Standard GA": (
        tools.cxTwoPoint,  # Two-point crossover swaps random segments of parent chromosomes.
        tools.mutFlipBit,  # FlipBit mutation toggles binary genes (0 or 1).
        tools.selTournament  # Tournament selection picks the best individuals from random subgroups.
    ),
    "Real-Coded GA": (
        partial(tools.cxBlend, alpha=0.5),  # Blend crossover mixes parent genes proportionally (alpha=0.5).
        partial(tools.mutGaussian, mu=0, sigma=1, indpb=0.2),  # Gaussian mutation perturbs genes with random noise.
        partial(tools.selTournament, tournsize=3)  # Tournament selection with group size 3.
    ),
    "NSGA-II": (
        partial(tools.cxSimulatedBinaryBounded, eta=20.0, low=LOWER_BOUND, up=UPPER_BOUND),
        # Simulated binary crossover within bounds.
        partial(tools.mutPolynomialBounded, eta=20.0, low=LOWER_BOUND, up=UPPER_BOUND, indpb=0.2),
        # Polynomial mutation with bounds.
        tools.selNSGA2  # NSGA-II selection keeps Pareto-optimal solutions for multi-objective optimization.
    ),
    "Hybrid GA": (
        tools.cxOnePoint,  # One-point crossover swaps genes up to a single random point.
        tools.mutShuffleIndexes,  # ShuffleIndexes mutation rearranges gene order.
        partial(tools.selTournament, tournsize=3)  # Tournament selection with group size 3.
    ),
}


# GA Feature Selection Process
def ga_feature_selection(evaluate_function, ga_method, n_generations=30, is_multi_objective=False):
    crossover, mutation, selection = ga_method
    n_features = X.shape[1]

    # Initialize evolution data for saving purposes
    evolution_data = {
        'total_generations': n_generations,
        'generations': [],
        'best': [],
        'avg': [],
        'worst': [],
        'ci_best': [],
        'ci_avg': [],
        'ci_worst': [],
        'feature_counts': defaultdict(int),
        'selected_features': {},
        'population_history': []
    }

    if ga_name == "NSGA-II":
        is_multi_objective = True

    # Select fitness and individual type
    if is_multi_objective:
        individual_type = creator.MultiObjectiveIndividual
    else:
        individual_type = creator.Individual

    # DEAP toolbox setup
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)

    def valid_individual():

        # Generate a valid individual with at least 3 features selected

        while True:
            individual = [random.randint(0, 1) for _ in range(n_features)]
            if sum(individual) >= 3:
                return individual_type(individual)

    def repair_individual(individual, min_features=3):

        # Ensures an individual has at least `min_features` selected.

        while sum(individual) < min_features:
            zero_indices = [i for i, value in enumerate(individual) if value == 0]
            if not zero_indices:
                break
            random_index = random.choice(zero_indices)
            individual[random_index] = 1
        return individual

    toolbox.register("individual", valid_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    if selection == tools.selTournament:
        toolbox.register("select", selection, tournsize=3)
    else:
        toolbox.register("select", selection)

    toolbox.register("mate", crossover)
    toolbox.register("mutate", mutation, indpb=0.2)
    toolbox.register("evaluate", evaluate_function)

    # Initialize population
    population = toolbox.population(n=50)
    for ind in population:
        repair_individual(ind)
        ind.fitness.values = toolbox.evaluate(ind)

    # Main GA loop
    for generation in range(n_generations):
        print(f"Generation {generation + 1}/{n_generations}")
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)

        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)

        # Fitness statistics
        fitnesses = [ind.fitness.values[0] for ind in population]
        ci_avg = compute_confidence_interval(fitnesses)
        ci_best = compute_confidence_interval(
            [ind.fitness.values[0] for ind in population if ind.fitness.values[0] == min(fitnesses)]
        )
        ci_worst = compute_confidence_interval(
            [ind.fitness.values[0] for ind in population if ind.fitness.values[0] == max(fitnesses)]
        )

        # Update evolution data
        evolution_data['generations'].append(generation)
        evolution_data['best'].append(min(fitnesses))
        evolution_data['avg'].append(np.mean(fitnesses))
        evolution_data['worst'].append(max(fitnesses))
        evolution_data['ci_best'].append(ci_best)
        evolution_data['ci_avg'].append(ci_avg)
        evolution_data['ci_worst'].append(ci_worst)
        evolution_data['population_history'].append(fitnesses)

        # Track feature selection
        generation_features = [
            [idx for idx, val in enumerate(ind) if val == 1] for ind in population
        ]
        evolution_data['selected_features'][generation] = generation_features

        # Update feature counts
        for ind in population:
            for idx, val in enumerate(ind):
                if val == 1:
                    evolution_data['feature_counts'][idx] += 1

        if is_multi_objective:
            population = toolbox.select(population + offspring,
                                        k=len(population))  # Combine parent and offspring for NSGA-II selection.
        else:
            population = toolbox.select(offspring,
                                        k=len(population))  # Select from offspring for single-objective optimization.

    # Return the best individual (NSGA-II: Pareto front)
    if is_multi_objective:
        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[
            0]  # Get the Pareto-optimal front.
        best_individual = tools.selBest(pareto_front, k=1)[0]  # Select the best individual from the Pareto front.
    else:
        best_individual = tools.selBest(population, k=1)[
            0]  # Select the best individual from the population for single-objective.

    selected_features = [index for index, val in enumerate(best_individual) if val == 1]
    return selected_features, evolution_data


# Run GA for all methods and compare
results = []
generations_list = [30, 40, 50]

for ga_name, ga_method in GA_METHODS.items():
    is_multi_objective = ga_name == "NSGA-II"
    for n_generations in generations_list:
        print(f"Processing {ga_name} for {n_generations} generations")
        selected_features, evolution_data = ga_feature_selection(
            evaluate_svm_nsga if is_multi_objective else evaluate_svm,
            ga_method,
            n_generations=n_generations,
            is_multi_objective=is_multi_objective
        )
        joblib.dump(evolution_data, f"results/{ga_name}_{n_generations}_evolution_data.joblib")
        print(f"Evolution data saved for {ga_name} with {n_generations} generations.")

        # Train SVM Model
        X_train_svm = X_train[:, selected_features]
        X_test_svm = X_test[:, selected_features]
        model = SVR(kernel='rbf')
        model.fit(X_train_svm, y_train)
        predictions = model.predict(X_test_svm)

        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        results.append({'GA Type': ga_name, 'Generations': n_generations, 'RMSE': rmse,
                        'Selected Features': len(selected_features)})

        # Plot Evolution with Confidence Intervals
        plt.figure(figsize=(10, 6))
        generations = range(len(evolution_data['best']))

        # Average fitness and its confidence interval
        plt.plot(generations, evolution_data['avg'], label='Average Fitness', color='blue')
        ci_lower_avg, ci_upper_avg = zip(*evolution_data['ci_avg'])
        plt.fill_between(generations, ci_lower_avg, ci_upper_avg, color='blue', alpha=0.2, label='95% CI Avg')

        # Best and worst fitness
        plt.plot(generations, evolution_data['best'], label='Best Fitness', color='green')
        plt.plot(generations, evolution_data['worst'], label='Worst Fitness', color='red')

        plt.title(f'{ga_name} Evolution with Confidence Intervals for {n_generations} Generations')
        plt.xlabel('Generations')
        plt.ylabel('Fitness (RMSE)')
        plt.legend()
        plt.grid()

        # Save Confidence Interval Plot
        ci_plot_filename = f"plots/{ga_name}_{n_generations}_confidence_intervals.png"
        plt.savefig(ci_plot_filename)
        plt.close()

        # Save Evolution Stats Plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(n_generations), evolution_data['best'], label='Best Fitness')
        plt.plot(range(n_generations), evolution_data['avg'], label='Average Fitness')
        plt.plot(range(n_generations), evolution_data['worst'], label='Worst Fitness')
        plt.title(f'{ga_name} Evolution Stats for {n_generations} Generations')
        plt.xlabel('Generations')
        plt.ylabel('Fitness (RMSE)')
        plt.legend()
        plt.grid()

        # Save plot
        evolution_stats_filename = f"plots/{ga_name}_{n_generations}_evolution_stats.png"
        plt.savefig(evolution_stats_filename)
        plt.close()

# Save Results as DataFrame
results_df = pd.DataFrame(results)
print(results_df)

# Plot Comparison of GA Types
for ga_name in GA_METHODS.keys():
    subset = results_df[results_df['GA Type'] == ga_name]
    plt.plot(subset['Generations'], subset['RMSE'], marker='o', linestyle='--', label=ga_name)

plt.title('Comparison of GA Types (RMSE vs. Generations)')
plt.xlabel('Number of Generations')
plt.ylabel('RMSE')
plt.legend()
plt.grid()
plt.show()
# Save the comparison plot
comparison_plot_filename = "plots/GA_Types_Comparison.png"
plt.savefig(comparison_plot_filename)
plt.close()

# Save results dataframe
results_df_filename = "results/results_dataframe.csv"
results_df.to_csv(results_df_filename, index=False)
