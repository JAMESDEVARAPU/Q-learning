import numpy as np

# Define the environment
class GridWorld:
    def __init__(self, grid_size, start, goal, obstacles):
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.state = start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 0:  # Up
            x = max(x - 1, 0)
        elif action == 1:  # Down
            x = min(x + 1, self.grid_size - 1)
        elif action == 2:  # Left
            y = max(y - 1, 0)
        elif action == 3:  # Right
            y = min(y + 1, self.grid_size - 1)

        # Check if the new state is an obstacle
        if (x, y) in self.obstacles:
            return self.state, -10, False  # Penalty for hitting an obstacle

        self.state = (x, y)

        # Check if the goal is reached
        if self.state == self.goal:
            return self.state, 10, True  # Reward for reaching the goal

        return self.state, -1, False  # Small penalty for each step

# Q-Learning Algorithm
def q_learning(env, episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    grid_size = env.grid_size
    q_table = np.zeros((grid_size, grid_size, 4))  # Q-table: (x, y, action)

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            # Choose action (epsilon-greedy)
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(0, 4)  # Random action
            else:
                action = np.argmax(q_table[state[0], state[1], :])  # Best action

            # Take action and observe the next state and reward
            next_state, reward, done = env.step(action)

            # Update Q-value using the Q-learning formula
            old_value = q_table[state[0], state[1], action]
            next_max = np.max(q_table[next_state[0], next_state[1], :])
            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            q_table[state[0], state[1], action] = new_value

            # Move to the next state
            state = next_state

    return q_table

# Main function
if __name__ == "__main__":
    # Define the grid world environment
    grid_size = 5
    start = (0, 0)
    goal = (4, 4)
    obstacles = [(1, 1), (2, 2), (3, 3)]

    env = GridWorld(grid_size, start, goal, obstacles)

    # Train the Q-learning agent
    q_table = q_learning(env, episodes=1000)

    # Print the learned Q-table
    print("Learned Q-Table:")
    print(q_table)

    # Test the learned policy
    state = env.reset()
    done = False
    print("\nTesting the learned policy:")
    while not done:
        action = np.argmax(q_table[state[0], state[1], :])
        next_state, reward, done = env.step(action)
        print(f"State: {state}, Action: {action}, Next State: {next_state}")
        state = next_state
