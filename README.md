# ReinforcementLearning
In this repository you will find implementations of well known reinforcement learning algorithms.

In the first commit I implement a deep q-network using pytorch, on the gym model CartPole-v1.
As you can see on the dqn_notebook, around after 70 episodes it starts finding a solution, although not super optimal (since it is trained only for 100 episodes).
Using more episodes to train the model will lead to better solutions (for 500 episodes, a stable solution can be found).
