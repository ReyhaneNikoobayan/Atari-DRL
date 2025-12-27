# Atari-DRL

## Overview

This repository contains **my implementations** of **Deep Reinforcement Learning (DRL)** algorithms for training agents on **Atari 2600 games**.  
I implemented the full training pipeline, including environment setup, agent learning, evaluation, and logging.

This project is intended for:
- Studying and experimenting with DRL algorithms
- Training agents on classic Atari benchmarks
- Reproducing and extending reinforcement learning research

The codebase is structured to allow easy integration of new algorithms, environments, and experimental settings.

# 🎮 Visual Advantage Actor–Critic (A2C) for Atari Pong

This project implements a **from-scratch Advantage Actor–Critic (A2C)** reinforcement learning algorithm trained on **Atari Pong (ALE/Pong-v5)** using **raw visual observations**.

---
📄 **source code:** 

**[A2C-pong](https://github.com/ReyhaneNikoobayan/Atari-DRL/blob/main/A2C-pong.ipynb)**

---
The agent learns directly from pixel inputs through convolutional neural networks and is optimized using policy-gradient methods. All components — environment vectorization, preprocessing, optimization, evaluation, and video recording — are implemented manually using **PyTorch** and **Gymnasium**, without relying on high-level RL training frameworks.

---

## 🧠 Algorithm: Advantage Actor–Critic (A2C)

Advantage Actor–Critic (A2C) is a **synchronous actor–critic algorithm** that learns a policy and a value function simultaneously.

The algorithm consists of two models:

- **Actor**: learns a stochastic policy π(a | s)
- **Critic**: estimates the state value V(s)

Training is performed using experiences collected from **multiple parallel environments**, and updates are applied synchronously for stability.

---

### Advantage Estimation

To reduce variance in policy-gradient updates, the actor is trained using an **advantage signal**.

Advantage at time t:

A(t) = Return(t) − Value(s_t)

### The return is computed using bootstrapping:

Return(t) = Reward(t) + γ × Return(t+1)


Where:
- γ is the discount factor
- Value(s_t) is the critic’s estimate of the current state

---

### Optimization Objectives

**Actor Objective**

The actor is optimized to increase the probability of actions that lead to higher-than-expected returns:

Actor Loss =
− log π(a_t | s_t) × Advantage(t)
− β × Entropy(π)


- The entropy term encourages exploration
- β controls the strength of entropy regularization

---

**Critic Objective**

The critic is trained to regress the predicted value toward the observed return:

Critic Loss =
0.5 × (Return(t) − Value(s_t))² 


---

## 🏗️ Network Architecture

Separate convolutional neural networks are used for the actor and critic.

### Input Representation

Observation shape: (Height=65, Width=84, Channels=4)


Four consecutive preprocessed frames are stacked to capture temporal information.

---

The agent uses **separate convolutional networks** for the actor and critic.

### Convolutional Encoder (Actor & Critic)

- 4 convolutional layers
- ReLU activations
- Strided convolutions for spatial downsampling

### Fully Connected Layers

- Hidden layer: 256 units
- **Actor output:** action logits
- **Critic output:** scalar state-value estimate

The convolution output size is computed dynamically using a dummy forward pass.

---

## Training Setup

- **Number of environments:** 8
- **Updates:** 15,000
- **Steps per update:** 128
- **Discount factor (γ):** 0.95
- **Actor learning rate:** 1e-4
- **Critic learning rate:** 1e-5
- **Entropy coefficient:** 0.01
- **Optimizer:** RMSprop
- **Device:** CPU / CUDA (if available)

---

## Training Process

During training, the agent:
1. Collects rollouts from parallel environments
2. Computes discounted returns
3. Calculates advantages
4. Updates actor and critic networks
5. Logs losses and episodic rewards

### Metrics Tracked

- Actor loss
- Critic loss
- Episodic reward
- Moving averages of training metrics

### Training Curves

The following plots are generated during training:
- Actor loss (moving average)
- Critic loss (moving average)
- Episodic reward (moving average over 50 episodes)

These plots help monitor learning stability and performance improvement over time.

---

## Evaluation / Testing

After training, the agent is evaluated in a separate environment with:
- Exploration disabled
- Greedy action selection (`argmax` over policy logits)
- Video recording enabled

### Test Setup

- **Test episodes:** 6
- **Render mode:** RGB frames
- **Frame stacking:** 4 frames
- **Videos saved to:** `videos/`

### Test Results

Each test episode reports the total episodic reward.

The trained agent achieves **optimal Pong performance**, consistently returning the ball, dominating rallies, and **attains maximum or near-maximum rewards**.

---

## Saving and Loading Models

The trained model is saved after training:

```bash
checkpoints/a2c_pong.pth

```

The trained A2C agent is saved as a PyTorch checkpoint containing both the actor and critic networks, along with their corresponding optimizers.

Saving Format:

     torch.save({"actor_conv": actor_conv.state_dict(),
    
    "actor_fc": actor_fc.state_dict(),
    
    "critic_conv": critic_conv.state_dict(),
    
    "critic_fc": critic_fc.state_dict(),
    
    "actor_optim": actor_optim.state_dict(),
    
    "critic_optim": critic_optim.state_dict(),}, path)

**[trained model](A2C-pong-model)**


## Training Results

The relatively high magnitude of the **actor loss** is expected, as it is computed using a sum over time steps rather than a mean. As a result, the loss value scales with episode length, leading to larger numerical values compared to formulations that use averaging.

In contrast, the **critic loss** is computed using the mean squared error, which normalizes the loss across time steps. As a result, the critic loss remains at a reasonable and stable magnitude, making the difference in scale between the two losses expected and not indicative of training instability.

### Episodic Reward

<img width="1694" height="908" alt="image" src="https://github.com/user-attachments/assets/8b6bc785-6ec5-4b5a-9991-4bb2fcd7fb2e" />

### Actor Loss

<img width="1700" height="760" alt="image" src="https://github.com/user-attachments/assets/fc45dbd6-b668-49c9-b4bc-f81670059296" />

### Critic Loss

<img width="1696" height="776" alt="image" src="https://github.com/user-attachments/assets/fbae26a8-5168-4298-ba96-0d1392f403aa" />



---

## Test Gameplay

The trained agent is evaluated in the test environment.
This folder contains six gameplay videos recorded during testing. You can download and view them directly from the links below.

**[test videos](test)**


Episode 1: Reward = 21.0

Episode 2: Reward = 19.0

Episode 3: Reward = 21.0

Episode 4: Reward = 21.0

Episode 5: Reward = 19.0

Episode 6: Reward = 20.0


