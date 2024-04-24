import numpy as np


class GridWorld:
    """Represents a Grid World environment for reinforcement learning tasks.

    Attributes:
        grid (list[list[str]]): Layout of the grid as a 2D array.
        size (int): Size of the grid.
        rewards (dict): Reward values for different cell types.
    """

    def __init__(self, grid_layout, reward_scenario):
        """Initialize the GridWorld environment.

        Args:
            grid_layout (list[list[str]]): Layout of the grid as a 2D array.
            reward_scenario (int): Scenario number for reward configuration.
        """
        self.grid = grid_layout
        self.size = len(grid_layout)
        self.set_rewards(reward_scenario)

    def set_rewards(self, reward_scenario):
        """Configure rewards based on the scenario.

        Args:
            reward_scenario (int): Specific scenario number to set different rewards.
        """
        self.rewards = {'S': 0, 'F': 0, 'H': -1, 'G': 10}
        if reward_scenario == 2:
            self.rewards['F'] = -0.1
        elif reward_scenario == 3:
            self.explored = set()  # Initialize explored set
        elif reward_scenario == 4:
            self.steps = 0  # Initialize step counter

    def get_reward(self, position):
        """Get the reward for the given position, adjusting for scenario specifics.

        Args:
            position (tuple): The grid coordinates (row, col).

        Returns:
            float: The reward value for the current position.
        """
        row, col = position
        cell_type = self.grid[row][col]
        reward = self.rewards[cell_type]

        if hasattr(self, 'steps'):
            self.steps += 1
            if cell_type == 'G':
                reward = max(reward - 0.1 * self.steps, 1)
        if hasattr(self, 'explored'):
            if position not in self.explored:
                self.explored.add(position)
                reward += 0.1

        return reward


class Agent:
    """Agent that can navigate through the GridWorld.

    Attributes:
        position (tuple): Current position of the agent in the grid.
    """

    def __init__(self, start_position):
        """Initialize the agent in the grid world.

        Args:
            start_position (tuple): Starting position of the agent in the grid.
        """
        self.position = start_position

    def move(self, grid, direction):
        """Move the agent in the specified direction within the grid boundaries.

        Args:
            grid (GridWorld): The grid world environment where the agent moves.
            direction (str): The direction to move ('up', 'down', 'left', 'right').
        """
        row, col = self.position
        moves = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
        dr, dc = moves.get(direction, (0, 0))
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < grid.size and 0 <= new_col < grid.size:
            self.position = (new_row, new_col)

    def stochastic_move(self, grid, action_probs):
        """Perform a stochastic move based on specified action probabilities.

        Args:
            grid (GridWorld): The grid world environment where the agent moves.
            action_probs (list[float]): Probabilities corresponding to each action.
        """
        direction = np.random.choice(['up', 'down', 'left', 'right'], p=action_probs)
        self.move(grid, direction)


def estimate_value_function(agent, grid, discount_factor=0.9, iterations=1000,
                            stochastic=False, action_probs=None):
    """Estimate the value function for the grid using the Bellman equation.

    Args:
        agent (Agent): The agent whose value function is to be estimated.
        grid (GridWorld): The grid world environment.
        discount_factor (float): The discount factor for future rewards.
        iterations (int): Number of iterations to perform for value estimation.
        stochastic (bool): If true, use stochastic transitions based on action_probs.
        action_probs (dict): Probabilities for taking each action if stochastic is True.

    Returns:
        numpy.ndarray: The estimated value function grid.
    """
    V = np.zeros((grid.size, grid.size))  # Initialize the value matrix

    for _ in range(iterations):
        last_iter_v = V.copy()
        for i in range(grid.size):
            for j in range(grid.size):
                if grid.grid[i][j] in ['H', 'G']:  # Handle terminal states
                    V[i, j] = round(grid.get_reward((i, j)), 4)
                    continue

                values = sum(
                    estimate_state_value(agent, grid, last_iter_v, i, j, direction, stochastic, action_probs, discount_factor)
                    for direction in ['up', 'down', 'left', 'right']
                )
                V[i, j] = values / 4  # Average the values from possible actions

    return V


def estimate_state_value(agent, grid, last_iter_v, i, j, direction, stochastic, action_probs, discount_factor):
    """Helper function to estimate the value for a state.

    Args:
        agent (Agent): The agent whose state value is estimated.
        grid (GridWorld): The grid world environment.
        last_iter_v (numpy.ndarray): The value matrix from the last iteration.
        i (int): Current row in the grid.
        j (int): Current column in the grid.
        direction (str): Direction to simulate the agent's move.
        stochastic (bool): If true, use stochastic transitions.
        action_probs (dict): Action probabilities if stochastic transition is used.

    Returns:
        float: The estimated value for the given state.
    """
    agent.position = (i, j)
    if stochastic:
        agent.stochastic_move(grid, action_probs[direction])
    else:
        agent.move(grid, direction)
    next_position = agent.position
    reward = grid.get_reward(next_position)
    if grid.grid[next_position[0]][next_position[1]] in ['H', 'G']:
        return reward
    return reward + discount_factor * last_iter_v[next_position]


def main():
    """Main function to set up and run value function estimation for various grid layouts and scenarios."""
    grid_layouts = [
        [['S', 'F', 'F'], ['F', 'H', 'F'], ['F', 'H', 'G']],
        [['S', 'F', 'F'], ['F', 'F', 'F'], ['F', 'F', 'G']],
        [['S', 'H', 'F'], ['F', 'F', 'H'], ['H', 'F', 'G']]
    ]
    reward_scenarios = [1, 2, 3, 4]
    action_probs = {
        'up': [0.8, 0, 0.1, 0.1],
        'down': [0, 0.7, 0.15, 0.15],
        'left': [0.3, 0.3, 0.4, 0],
        'right': [0.3, 0.1, 0, 0.6]
    }

    for grid_layout in grid_layouts:
        for reward_scenario in reward_scenarios:
            print(f"Testing grid layout: {grid_layout} with reward scenario {reward_scenario}")
            world = GridWorld(grid_layout, reward_scenario)
            agent = Agent(start_position=(0, 0))
            V_deterministic = estimate_value_function(agent, world, stochastic=False)
            world = GridWorld(grid_layout, reward_scenario)
            agent = Agent(start_position=(0, 0))
            V_stochastic = estimate_value_function(agent, world, stochastic=True, action_probs=action_probs)
            print("Deterministic Value Function V(s):")
            print(V_deterministic)
            print("Stochastic Value Function V(s):")
            print(V_stochastic)


if __name__ == "__main__":
    main()
