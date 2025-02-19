import random
import numpy as np
import math
import matplotlib.pyplot as plt
from graph import Graph
from graph_algorithms import get_diameter, get_clustering_coefficient, get_degree_distribution
from scipy.stats import linregress
from tqdm import tqdm  
import matplotlib.ticker as ticker

# Function to generate an Erdos-Renyi random graph
def generate_erdos_renyi_graph(n, p):
    edges = set()
    v,w = 1,-1
    while v < n:
        r = random.random()
        w = w + 1 + int(math.log2(1 - r) / math.log(1 - p))
        while w >= v and v < n:
            w = w-v
            v = v+1
        if v < n:
            edges.add((v, w))
    return Graph(n, edges)

# Function to generate a Barabasi-Albert random graph

import random
from graph import Graph  # Importing the Graph class from the graph.py file

def generate_barabasi_albert_graph(n: int, degree: int) -> Graph:
    # Initialize an empty list for edges
    M = []
    # Step 1: Generate an initial set of edges
    for v in range(n):
        for i in range(degree):
            M.append(v)
        
        # Randomly choose a previous edge and update the list M
        for i in range(degree):
            r = random.randint(0, len(M) - 1)
            M.append(M[r])

    # Step 2: Create the set of edges E
    edges = set()
    for i in range(0, len(M), 2):
        u, v = M[i], M[i + 1]
        if u != v:
            edges.add((u, v))  # Avoid self-loops

    # Now create the graph with the generated edges
    return Graph(n, edges)

'''def generate_barabasi_albert_graph(n, m):
    M1 = [0] * (2 * n * m) 
    for v in range(n):
        for i in range(m):
            M1[2 * (v * m + i)] = v
    for v in range(n):
        for i in range(m):
            r = random.randint(0, 2 * (v * m + i))
            M1[2 * (v * m + i) + 1] = M1[r]
    edges = set()
    for i in range(n * m):
        edge = (min(M1[2 * i], M1[2 * i + 1]), max(M1[2 * i],
                                                 M1[2 * i + 1]))
        edges.add(edge)
    return Graph(n, edges)'''

# Function to generate graph and calculate properties
def run_experiment(n, graph_type='erdos_renyi', param=0.1):
    if graph_type == 'erdos_renyi':
        p = 2*math.log(n) / n  # Adjust the probability based on graph size
        graph = generate_erdos_renyi_graph(n, p)
    else:  # Barabasi-Albert
        m = 5  # Number of neighbors to attach at each step
        graph = generate_barabasi_albert_graph(n, m)
    
    # Get the graph properties
    diameter = get_diameter(graph)
    clustering_coefficient = get_clustering_coefficient(graph)
    degree_distribution = get_degree_distribution(graph)
    
    return diameter, clustering_coefficient, degree_distribution



# Function to add the best fit line equation in the correct form
def add_best_fit_line(x, y, label='Best Fit Line', log_scale=False):
    # Apply log transformation for log-log scale
    if log_scale:
        log_x = np.log(x)
        log_y = np.log(y)
        
        # Perform linear regression in log-log space
        slope, intercept, _, _, _ = linregress(log_x, log_y)
        
        # Equation for log-log plot: y = m * log(x) + c
        equation = f'{label}: y = {slope:.2f}*log(x) + {intercept:.2f}'
        best_fit_line = np.exp(intercept) * (x ** slope)  # y = exp(c) * x^m (in log-log scale)
        
    else:
        # Perform linear regression in lin-log space
        slope, intercept, _, _, _ = linregress(x, np.log(y))
        
        # Equation for lin-log plot: y = m * x + c
        equation = f'{label}: y = {slope:.2f}x + {intercept:.2f}'
        best_fit_line = np.exp(intercept) * (x ** slope)  # y = c * x^m (in lin-log scale)
    
    return best_fit_line, equation, slope

def plot_graph_properties(sizes, avg_diameters, avg_clustering_coefficients, graph_type):
    # Plotting average diameter vs. size on lin-log scale
    plt.figure(figsize=(8, 6))
    plt.scatter(sizes, avg_diameters, color='b', label='Average Diameter', marker='o')
    plt.plot(sizes, avg_diameters, color='b', linestyle='-', alpha=0.5)  # Connecting the points with a line
    plt.xscale('linear')
    plt.yscale('log')
    
    plt.xlabel('Number of nodes')
    plt.ylabel('Diameter')
    plt.title(f'Average Diameter vs. Number of nodes ({graph_type})')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plotting average clustering coefficient vs. size on lin-log scale
    plt.figure(figsize=(8, 6))
    plt.scatter(sizes, avg_clustering_coefficients, color='b', label='Average Clustering Coefficient', marker='o')
    plt.plot(sizes, avg_clustering_coefficients, color='b', linestyle='-', alpha=0.5)  # Connecting the points with a line
    plt.xscale('linear')
    plt.yscale('log')
    
    plt.xlabel('Number of nodes')
    plt.ylabel('Clustering Coefficient')
    plt.title(f'Average Clustering Coefficient vs. Number of nodes ({graph_type})')
    plt.grid(True)
    plt.legend()
    plt.show()



# Function to plot degree distribution on lin-lin and log-log scales
def plot_degree_distribution(degree_distribution, graph_type, size):
    # Sort degree distribution by degree (x-axis)
    degrees = sorted(degree_distribution.keys())
    counts = [degree_distribution[degree] for degree in degrees]
    
    # Plot degree distribution on lin-lin scale
    plt.figure(figsize=(8, 6))
    plt.scatter(degrees, counts, color='b', label='Degree Distribution', marker='o')

    # Add best fit line on lin-lin scale
    best_fit_line, equation, slope = add_best_fit_line(degrees, counts, label='Best Fit Line (Lin-Lin)', log_scale=False)
    print(f"Slope for degree distribution (Lin-Lin): {slope}")  # Print the slope for degree distribution (lin-lin)
    plt.plot(degrees, best_fit_line, label=equation, linestyle='--', color='r')
    
    # Increase y-axis ticks
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune='lower', nbins=6))

    plt.xlabel('Degree')
    plt.ylabel('Count')
    plt.title(f'Degree Distribution (Lin-Lin scale) - {graph_type}, Size {size}')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot degree distribution on log-log scale
    plt.figure(figsize=(8, 6))
    plt.scatter(degrees, counts, color='b', label='Degree Distribution', marker='o')

    # Apply logarithmic transformation for log-log scale (only if degree > 0)
    log_degrees = np.log(degrees)
    log_counts = np.log(counts)

    # Add best fit line on log-log scale
    best_fit_line, equation, slope = add_best_fit_line(degrees, counts, label='Best Fit Line (Log-Log)', log_scale=True)
    print(f"Slope for degree distribution (Log-Log): {slope}")  # Print the slope for degree distribution (log-log)
    plt.plot(degrees, best_fit_line, label=equation, linestyle='--', color='r')

    # Increase y-axis ticks for the log-log scale
    ax = plt.gca()  # Get current axes
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune='lower', nbins=6))

    # Apply log scale for both axes
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Degree (log scale)')
    plt.ylabel('Count (log scale)')
    plt.title(f'Degree Distribution (Log-Log scale) - {graph_type}, Size {size}')
    plt.grid(True)
    plt.legend()
    plt.show()

    return slope  # Return slope for power-law exponent

# Function to run and plot experiments for diameter, clustering coefficient, and degree distribution
def run_and_plot():
    # Diameter and clustering coefficient experiment
    sizes = [500, 1000, 5000, 10000, 25000, 50000, 100000]  # Different graph sizes (small, medium, large)
    avg_diameters = []
    avg_clustering_coefficients = []
    degree_distributions = []

    # Graph parameters (adjust based on student ID)
    graph_type = 'erdos_renyi'  # or 'barabasi_albert' 
    param = 0.1  # For Erdos-Renyi, adjust probability; for Barabasi-Albert, adjust m
    num_trials = 5  # Number of trials per graph size to average the results

    for n in tqdm(sizes, desc="Processing graph sizes"):  # Add tqdm progress bar for graph sizes
        diameters = []
        clustering_coefficients = []
        degree_distribution = None

        # Run the experiment multiple times to compute average values
        for _ in tqdm(range(num_trials), desc=f"Running trials for size {n}", leave=False):  # Add progress bar for trials
            diameter, clustering_coefficient, degree_distribution = run_experiment(n, graph_type, param)
            diameters.append(diameter)
            clustering_coefficients.append(clustering_coefficient)
        
        # Compute averages for diameter and clustering coefficient
        avg_diameters.append(np.mean(diameters))
        avg_clustering_coefficients.append(np.mean(clustering_coefficients))
        degree_distributions.append(degree_distribution)
        
    # Plotting average diameter and clustering coefficient
    plot_graph_properties(sizes, avg_diameters, avg_clustering_coefficients, graph_type)

    # Degree distribution analysis for graph sizes 1,000, 10,000, and 100,000
    degree_slope = []
    for size in [1000, 10000, 100000]:
        degree_distribution = run_experiment(size, graph_type, param)[2]
        slope = plot_degree_distribution(degree_distribution, graph_type, size)
        degree_slope.append(slope)
        
        # Print the slope for power law determination
        print(f"Slope for size {size}: {slope}")
        
# Running the experiment and plotting diameter, clustering coefficient, and degree distribution
run_and_plot()
