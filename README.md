
# Random Graph Generation and Analysis

This project implements the generation and analysis of random graphs, specifically focusing on the **Erdős–Rényi** model and the **Barabási–Albert** model. It provides functions for graph generation, analysis of graph properties (such as clustering coefficient, diameter, and degree distribution), and visualizations to explore the behavior of these random graphs.

## Features

- **Graph Generation**:
  - Erdős–Rényi random graph (`generate_erdos_renyi_graph`).
  - Barabási–Albert scale-free graph (`generate_barabasi_albert_graph`).

- **Graph Property Calculations**:
  - **Diameter**: Longest shortest path between any two nodes.
  - **Clustering Coefficient**: Measures the degree of clustering in a graph (i.e., how often neighbors of a node are neighbors themselves).
  - **Degree Distribution**: Distribution of node degrees (number of connections).

- **Experiments and Plots**:
  - Diameter and clustering coefficient vs. number of nodes.
  - Degree distribution (linear and log-log scales) with power-law fitting.
  - Visualization of graph properties using **Matplotlib**.

- **Best Fit Line for Degree Distribution**:
  - Adds a regression line to the degree distribution plot and calculates the power-law exponent.

## Installation

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/your-username/random-graph-analysis.git
   ```
2. Navigate to the project directory:
   ```bash
   cd random-graph-analysis
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the experiments and visualize the graph properties, execute the following Python script:

```bash
python random_graph_analysis.py
```

### Experiment Parameters:
- **`n`**: Number of nodes in the graph.
- **`p`**: Probability of an edge between any two nodes (for Erdős–Rényi graph).
- **`degree`**: Number of edges each new node connects to (for Barabási–Albert graph).

The script will:
1. Generate a graph based on the selected model.
2. Calculate and plot the graph properties (diameter, clustering coefficient, and degree distribution).
3. Display the results in a set of plots.

## Functions

### `generate_erdos_renyi_graph(n, p)`
Generates an Erdős–Rényi random graph with **n** nodes and probability **p** for edge creation.

### `generate_barabasi_albert_graph(n, degree)`
Generates a Barabási–Albert scale-free graph with **n** nodes and each new node attaching to **degree** existing nodes.

### `run_experiment(n, graph_type, param)`
Runs the graph generation and analysis for a given graph type and parameters, collecting graph properties like diameter, clustering coefficient, and degree distribution.

### `run_and_plot()`
Runs experiments for multiple graph sizes, collects data for diameter and clustering coefficient, and generates plots.

### `plot_graph_properties()`
Plots the **diameter** and **clustering coefficient** as a function of the number of nodes (with a log scale on the x-axis).

### `plot_degree_distribution()`
Plots the **degree distribution** of the generated graph, including a log-log scale and a fitted power-law curve.

### `add_best_fit_line()`
Calculates and adds a best-fit line (regression) to the degree distribution plot.

## Dependencies

- `matplotlib`: For plotting the graphs and visualizing data.
- `networkx`: For graph generation and manipulation.
- `numpy`: For numerical calculations.
- `scipy`: For regression and statistical analysis.

## Contact

For any questions or issues, feel free to reach out via GitHub Issues or email **[bhuvanchandra3008@gmail.com](mailto:bhuvanchandra3008@gmail.com)**.
