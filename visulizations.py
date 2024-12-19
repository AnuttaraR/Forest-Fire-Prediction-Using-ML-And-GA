# Visualization from Saved SVM Models and GA Evolution Data
# ---------------------------------------------------------
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter
from collections import Counter

# Paths to saved models and GA evolution data
results_save_path = "svm_results.joblib"
models_save_path = "svm_models.joblib"
evolution_data_save_path = "all_ga_evolution_data.joblib"
cleaned_data_path = "cleaned_forestfires.csv"

# Load data
svm_results = joblib.load(results_save_path)
svm_models = joblib.load(models_save_path)
all_ga_evolution_data = joblib.load(evolution_data_save_path)

print("SVM Results, Models, and GA Evolution Data Loaded Successfully")

# Load dataset for feature reference
df = pd.read_csv(cleaned_data_path)
X = df.drop(columns=['log_area'])
y = df['log_area']

# Visualization: Fitness Evolution Across GA Methods
for method, ga_data in all_ga_evolution_data.items():
    generations = ga_data['generations']
    best_fitness = ga_data['best']
    avg_fitness = ga_data['avg']
    worst_fitness = ga_data['worst']

    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_fitness, label='Best Fitness', color='green', linewidth=2)
    plt.plot(generations, avg_fitness, label='Average Fitness', color='orange', linestyle='--')
    plt.plot(generations, worst_fitness, label='Worst Fitness', color='red', linestyle='-.')
    plt.title(f'Fitness Evolution - {method}')
    plt.xlabel('Generation')
    plt.ylabel('Fitness (RMSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Visualization: Feature Selection Frequency Across GA Methods
for method, ga_data in all_ga_evolution_data.items():
    feature_counts = ga_data['feature_counts']
    features = list(range(X.shape[1]))  # Total feature indices
    counts = [feature_counts.get(i, 0) for i in features]

    plt.figure(figsize=(10, 6))
    plt.bar(features, counts, color='skyblue')
    plt.title(f'Feature Selection Frequency - {method}')
    plt.xlabel('Feature Index')
    plt.ylabel('Selection Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Visualization: Top 10 Most Frequently Selected Features
for method, ga_data in all_ga_evolution_data.items():
    feature_counts = ga_data['feature_counts']
    sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    top_features, top_counts = zip(*sorted_features)

    plt.figure(figsize=(8, 6))
    plt.bar(top_features, top_counts, color='darkblue')
    plt.title(f'Top 10 Most Frequently Selected Features - {method}')
    plt.xlabel('Feature Index')
    plt.ylabel('Selection Count')
    plt.tight_layout()
    plt.show()

# Visualization: Fitness Evolution with Confidence Intervals
for method, ga_data in all_ga_evolution_data.items():
    generations = ga_data['generations']
    population_history = ga_data['population_history']

    avg_fitness = [np.mean(gen) for gen in population_history]
    std_fitness = [np.std(gen) for gen in population_history]
    best_fitness = ga_data['best']

    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_fitness, label='Best Fitness', color='green')
    plt.plot(generations, avg_fitness, label='Average Fitness', color='orange')
    plt.fill_between(generations,
                     np.array(avg_fitness) - np.array(std_fitness),
                     np.array(avg_fitness) + np.array(std_fitness),
                     color='orange', alpha=0.3, label='Confidence Interval')
    plt.title(f'Fitness Evolution with Confidence Interval - {method}')
    plt.xlabel('Generation')
    plt.ylabel('Fitness (RMSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Animated Visualization: Fitness Over Generations (Saved as GIF)
for method, ga_data in all_ga_evolution_data.items():
    population_history = ga_data['population_history']

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f'Population Fitness Evolution (Worst-to-Best) - {method}')
    ax.set_xlim(0, len(population_history[0]))
    ax.set_ylim(min(min(population_history)), max(max(population_history)))
    ax.set_xlabel('Sorted Individuals')
    ax.set_ylabel('Fitness (RMSE)')

    line, = ax.plot([], [], 'bo')

    def update(frame):
        # Sort the fitness values for the current generation
        sorted_fitness = sorted(population_history[frame], reverse=True)  # Sort worst-to-best
        line.set_data(range(len(sorted_fitness)), sorted_fitness)
        return line,

    ani = FuncAnimation(fig, update, frames=len(population_history), interval=500, blit=True)
    gif_path = f'fitness_evolution_{method.lower().replace(" ", "_")}_sorted.gif'
    ani.save(gif_path, writer=PillowWriter(fps=2))
    print(f"Animated Fitness Plot Saved (Sorted): {gif_path}")
    plt.close(fig)

print("All Visualizations Complete.")
