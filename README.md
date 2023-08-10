[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# DRLND - Project 2: Continuous Control

### Project Details

Welcome to the project repository! This README provides instructions on how to use this repository to train an agent to control a double-jointed arm in the Reacher environment.

![Trained Agent][image1]

In this environment, you have a double-jointed arm that can move to target locations. The goal of your agent is to maintain its position at the target location for as many time steps as possible. For each step that the agent's hand is in the goal location, a reward of +0.1 is provided.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.


## Getting Started

To get started with this repository, follow the instructions below to install the necessary dependencies and set up the project environment.

### Prerequisites

- Python 3.6.13
- PyTorch 0.4.0
- Unityagents 0.4.0

### Installation

1. Clone the repository to your local machine using the following command:

   ```
   git clone https://github.com/HasarinduPerera/drlnd-continuous-control-project
   ```

2. Change into the project directory:

   ```
   cd drlnd-continuous-control-project
   ```

3. Start the project without any additional work as the required environment, "Banana.app," is already uploaded in this project.

## Instructions

To train and test the agent, follow the instructions below.

1. Make sure you have completed the installation steps mentioned above.

2. Open the `Continuous_Control.ipynb` notebook. It serves as the entry point for the project and contains two modes: one for training and one for testing.

3. If you already have a pre-trained model, make sure you have the `checkpoint.pth` file in your project directory. This file saves the weights of the trained model.

4. If you want to train the DQN-Agent, run the training mode in the `Continuous_Control.ipynb` notebook. This will train the agent using reinforcement learning techniques.

5. If you only want to test the agent using a pre-trained model, load the `checkpoint.pth` file and start the test mode in the `Continuous_Control.ipynb` notebook. This will evaluate the agent's performance in the environment.

Alternatively, you can use the `Continuous_Control.py` file if you prefer not to use a Jupyter Notebook. It contains the same code as in the `Continuous_Control.ipynb` notebook.

Congratulations! You have successfully trained and tested the agent in the project environment. Feel free to explore the code, experiment with different configurations, and adapt it to your specific requirements.

If you have any questions or encounter any issues while using this repository, please don't hesitate to open an issue.
