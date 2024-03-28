# Home Energy Management System (HEMS) Optimization

## Overview

This project implements a Home Energy Management System (HEMS) optimization using Linear Programming (LP). It's designed for homeowners, energy managers, or anyone interested in optimizing household energy consumption and costs.

## Features

The HEMS optimization model optimizes energy purchase, sale, and battery usage to minimize the total cost of energy consumption. The key features include:

- **Battery Management:**
  - Considers battery capacity, maximum charge rate, and maximum discharge rate
  - Optimizes battery usage based on energy prices and consumption patterns
- **Solar Energy Integration:**
  - Incorporates solar energy production into the optimization model
  - Utilizes excess solar energy to charge the battery or sell back to the grid
- **Energy Purchase and Sale:**
  - Optimizes energy purchase from the grid during low-price periods
  - Sells excess energy back to the grid during high-price periods

The optimization model aims to minimize the total cost of energy consumption while satisfying the household's energy demand and considering the available solar energy and battery storage.

## Getting Started

### Prerequisites

- Python 3.x
- PuLP
- Matplotlib
- NumPy

You can install these dependencies with `pip install -r requirements.txt`.

### Usage

To use the script, run:
python hems_optimization.py

The script will solve the optimization model, print the results, and display the visualizations.

## Results

The optimization model provides the following results:

- Optimal energy purchase and sale at each time step
- Battery state at each time step
- Total costs, revenue, and profit based on the optimal solution

The results are printed in the console and visualized using Matplotlib. The visualizations include:

- Battery charge over time
- Energy purchase and sale over time, along with the corresponding prices
- Solar energy production and household consumption over time

These results can be used to gain insights into the optimal energy management strategy for the household, considering the available solar energy, household consumption patterns, and energy prices.

## Contributing

Contributions to the project are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the GitHub repository.
