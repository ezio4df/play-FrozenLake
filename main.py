# This is a sample Python script.

# Press Ctrl+F5 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press F9 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/





# FrozenLake 8x8

# class FrozenLakeV1(nn.Module):
#     def __init__(self, state_size, action_size, hidden_size=64):
#         super(FrozenLakeV1, self).__init__()
#         self.fc1 = nn.Linear(state_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, action_size)
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)


agent, env = train_dqn(modelClass=FrozenLakeV0, is_slippery=False, episodes=2000)


# Evaluate
eval_env = gym.make("FrozenLake8x8-v1", is_slippery=False, render_mode="human")
evaluate_agent(agent, eval_env, episodes=2)
eval_env.close()