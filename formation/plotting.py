import matplotlib.pyplot as plt

# Data
num_robots = [2, 3, 4, 5, 6]
iteration_time = [0.09, 0.22, 0.45, 1.1, 2.3]

# Plot
plt.plot(num_robots, iteration_time, marker='o', linestyle='-')
plt.xlabel('Number of Robots')
plt.ylabel('Iteration Time (seconds)')
plt.xticks(num_robots)  # Set x-ticks to only display the number of robots
plt.grid(False)
plt.show()
