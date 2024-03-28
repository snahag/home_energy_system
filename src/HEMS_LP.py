import pulp
import matplotlib.pyplot as plt
import numpy as np

class HEMS_LP:
    def __init__(self, time_steps, max_battery_capacity, max_charge_rate, max_discharge_rate, initial_battery_state,
                 solar_production, house_consumption, purchase_price, sell_price):
        # Initialize parameters
        self.time_steps = time_steps
        self.max_battery_capacity = max_battery_capacity
        self.max_charge_rate = max_charge_rate
        self.max_discharge_rate = max_discharge_rate
        self.initial_battery_state = initial_battery_state
        self.solar_production = solar_production
        self.house_consumption = house_consumption
        self.purchase_price = purchase_price
        self.sell_price = sell_price
        
        # Create optimization model
        self.model = pulp.LpProblem("HEMS_LP", pulp.LpMinimize)
        
        # Create decision variables
        self.create_variables()

    def create_variables(self):
        # Decision variables for energy purchase, sell, and battery state
        self.energy_purchase = pulp.LpVariable.dicts("Energy_purchase", (t for t in range(self.time_steps)), lowBound=0)
        self.energy_sell = pulp.LpVariable.dicts("Energy_sell", (t for t in range(self.time_steps)), lowBound=0)
        self.battery_state = pulp.LpVariable.dicts("Battery_state", (t for t in range(self.time_steps+1)), lowBound=0, upBound=self.max_battery_capacity)

    def create_objective_function(self):
        # Objective function: minimize cost
        self.model += pulp.lpSum([self.purchase_price(t) * self.energy_purchase[t] - self.sell_price(t) * self.energy_sell[t] - ((self.sell_price(t))*0.01 * self.battery_state[t]) for t in range(self.time_steps)])

    def create_battery_balance_constraints(self):
        # Battery balance constraints
        self.model += self.battery_state[0] == self.initial_battery_state
        for t in range(self.time_steps):
            self.model += self.battery_state[t+1] == self.battery_state[t] + self.energy_purchase[t] - self.energy_sell[t] + self.solar_production(t) - self.house_consumption(t)

    def create_max_charge_discharge_constraints(self):
        # Maximum charge and discharge rate constraints
        for t in range(self.time_steps-1):
            self.model += self.battery_state[t+1] - self.battery_state[t] <= self.max_charge_rate
            self.model += self.battery_state[t] - self.battery_state[t+1] <= self.max_discharge_rate

    def create_battery_capacity_constraints(self):
        # Battery capacity constraints
        for t in range(self.time_steps+1):
            self.model += self.battery_state[t] <= self.max_battery_capacity
            self.model += self.battery_state[t] >= 0

    def create_penalty_variables(self, penalty_cost_per_kWh):
        # Penalty variables and constraints
        self.penalty = pulp.LpVariable.dicts("Penalty", (t for t in range(self.time_steps)), lowBound=0)
        self.model += pulp.lpSum([self.penalty[t] for t in range(self.time_steps)])

        for t in range(self.time_steps):
            #self.model += self.penalty[t] >= penalty_cost_per_kWh * (self.max_battery_capacity/2 - self.battery_state[t])
            self.model += self.penalty[t] >= penalty_cost_per_kWh * pulp.lpSum([self.max_battery_capacity/2 - self.battery_state[t], 0])
            self.model += self.penalty[t] >= 0
    
    # def create_penalty_variables(self, penalty_cost_per_kWh):
    #     # Penalty variables and constraints
    #     self.penalty = pulp.LpVariable.dicts("Penalty", (t for t in range(self.time_steps)), lowBound=0)
    #     self.model += pulp.lpSum([self.penalty[t] for t in range(self.time_steps)])

    #     for t in range(self.time_steps):
    #         # Binary variable to activate penalty when battery state is below max_battery_capacity/2
    #         below_threshold = pulp.LpVariable(f"Below_threshold_{t}", cat="Binary")

    #         # Constraint to activate penalty only when battery state is below max_battery_capacity/2
    #         self.model += self.penalty[t] >= penalty_cost_per_kWh * (self.max_battery_capacity/2 - self.battery_state[t]) - self.max_battery_capacity * (1 - below_threshold)
    #         self.model += self.penalty[t] <= penalty_cost_per_kWh * (self.max_battery_capacity/2 - self.battery_state[t]) + self.max_battery_capacity * below_threshold

    #         # Constraint to set the binary variable based on battery state
    #         self.model += self.battery_state[t] <= self.max_battery_capacity/2 + self.max_battery_capacity * (1 - below_threshold)
    #         self.model += self.battery_state[t] >= self.max_battery_capacity/2 * below_threshold

    #         self.model += self.penalty[t] >= 0

    def solve(self, penalty_cost_per_kWh=None):
        # Create objective function and constraints
        self.create_objective_function()
        self.create_battery_balance_constraints()
        self.create_max_charge_discharge_constraints()
        self.create_battery_capacity_constraints()
        
        # Add penalty variables and constraints if penalty_cost_per_kWh is provided
        if penalty_cost_per_kWh is not None:
            self.create_penalty_variables(penalty_cost_per_kWh)
        
        # Solve the model
        self.model.solve()
        
        # Extract the optimal values of decision variables
        energy_purchase = [self.energy_purchase[t].varValue for t in range(self.time_steps)]
        energy_sell = [self.energy_sell[t].varValue for t in range(self.time_steps)]
        battery_state = [self.battery_state[t].varValue for t in range(self.time_steps + 1)]
        
        return energy_purchase, energy_sell, battery_state

def solar_production(t):
    # Example: Maximum production at noon, no production at night
    return max(0, 20 - abs(t - 48))/100 if 0 <= t <= 95 else 0 

def house_consumption(t):
    # Example: Higher consumption in the morning and evening
    return (0.15 if 20 <= t <= 40 or 60 <= t <= 80 else 0.07) if 0 <= t <= 95 else 0

def purchase_price(t):
    # Example: Higher prices during peak load times
    return (0.30 if 23 <= t <=  87 else 0.27) if 0 <= t <= 95 else 0 

def sell_price(t):
    # Example: Lower prices during peak load times
    return (0.25 if 23 <= t <= 87 else 0.20) if 0 <= t <= 95 else 0

def main():
    # Define problem parameters
    max_battery_capacity = 20  # Maximum capacity in kWh
    max_charge_rate = 4 * 0.1  # Maximum charge rate per 15 minutes in kWh
    max_discharge_rate = 3.7 * 0.1  # Maximum discharge rate per 15 minutes in kWh
    initial_battery_state = 7  # Initial battery state in kWh
    penalty_cost_per_kWh = None # Penalty for minimum
    T = 96

    # Create an instance of the HEMS_LP model
    model = HEMS_LP(T, max_battery_capacity, max_charge_rate, max_discharge_rate, initial_battery_state, 
                    solar_production, house_consumption, purchase_price, sell_price)

    # Solve the model
    energy_purchase, energy_sell, battery_state = model.solve(penalty_cost_per_kWh)

    # Calculate total costs, revenue, and profit
    total_costs = sum([purchase_price(t) * energy_purchase[t] for t in range(T)])
    total_revenue = sum([sell_price(t) * energy_sell[t] for t in range(T)])
    total_profit = total_revenue - total_costs

    # Store solar energy and house energy data
    solar_energy = {t: solar_production(t) for t in range(T)}
    house_energy = {t: house_consumption(t) for t in range(T)}

    # Print results
    for t in range(T):
        print(f"Time step {t}: Energy_purchase = {energy_purchase[t]}, Energy_sell = {energy_sell[t]}, Battery_state = {battery_state[t]}, Solar_energy = {solar_energy[t]}, House_energy = {house_energy[t]}")

    print(f"Total_costs: {total_costs} €, Total_revenue: {total_revenue} €, Total_profit: {total_profit} €")

    # Plot the results
    plot_results(T, battery_state, energy_purchase, energy_sell, solar_energy, house_energy)

def plot_results(T, battery_state, energy_purchase, energy_sell, solar_energy, house_energy):
    intervalle = np.arange(T)

    # Create a single figure for all subplots
    plt.figure(figsize=(10, 10))

    # Subplot for the battery
    plt.subplot(3, 1, 1)  # 3 rows, 1 column, 1st plot
    plt.plot(intervalle, battery_state[:-1], label='Battery_state', color='red')  # Exclude the last value
    plt.xlabel('Time intervals (15 minutes)')
    plt.ylabel('Battery charge (kWh)')
    plt.ylim(-2, 25)
    plt.title('Battery Charge Over 96 Time Intervals')
    plt.legend()

    # Subplot for energy purchase and sell
    plt.subplot(3, 1, 2)  # 3 rows, 1 column, 2nd plot
    plt.plot(intervalle, energy_purchase, label='Energy_purchase', color='blue')
    plt.plot(intervalle, energy_sell, label='Energy_sell', color='green')
    plt.ylabel('Amount of Energy (kWh)')

    # Create a secondary y-axis for prices
    ax2 = plt.gca().twinx()
    ax2.plot(intervalle, [purchase_price(t) for t in range(T)], label='Purchase_price (€/kWh)', color='blue', linestyle='dashed')
    ax2.plot(intervalle, [sell_price(t) for t in range(T)], label='Sell_price (€/kWh)', color='green', linestyle='dashed')
    ax2.set_ylabel('Price (€/kWh)')

    plt.xlabel('Time intervals (15 minutes)')
    plt.title('Energy Purchase and Sale Over 96 Time Intervals')
    plt.legend(loc='upper left')

    # Subplot for solar energy and house energy
    plt.subplot(3, 1, 3) 
    plt.plot(intervalle, [solar_energy[t] for t in range(T)], label='Solar_energy', color='yellow')
    plt.plot(intervalle, [house_energy[t] for t in range(T)], label='House_energy', color='purple')
    plt.xlabel('Time intervals (15 minutes)')
    plt.ylabel('Amount of Energy (kWh)')
    plt.title('Solar Energy Production and Household Consumption Over 96 Time Intervals')
    plt.legend()

    # Display the figure with all subplots
    plt.tight_layout()  
    plt.show()

if __name__ == "__main__":
    main()
