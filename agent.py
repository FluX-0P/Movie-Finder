import numpy as np
import pandas as pd

class RecommendationAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9):
        self.actions = actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.q_table = pd.DataFrame(columns=actions, dtype=np.float64)

    def update_q_value(self, state, action, reward, next_state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))

        current_q = self.q_table.at[state, action]
        max_next_q = self.q_table.loc[next_state].max() if next_state in self.q_table.index else 0
        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_next_q)
        self.q_table.at[state, action] = new_q
