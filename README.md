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

The agent learns directly from pixel inputs through convolutional neural networks and is optimized using policy-gradient methods. All components — environment vectorization, preprocessing, optimization, evaluation, and video recording — are implemented manually using **PyTorch** and **Gymnasium**, without relying on high-level RL training frameworks.

---

## 🎯 Project Motivation

This project was developed to gain a deep, practical understanding of:

- Actor–critic reinforcement learning algorithms
- Policy-gradient optimization with value-function baselines
- End-to-end visual learning using convolutional networks
- Parallel environment training for improved sample efficiency

Atari Pong serves as a controlled benchmark for studying learning dynamics in visual reinforcement learning.

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

# The return is computed using bootstrapping:

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

### Actor Network

Conv2D: 32 filters, kernel=8, stride=4
ReLU
Conv2D: 32 filters, kernel=4, stride=2
ReLU
Conv2D: 32 filters, kernel=4, stride=2
ReLU
Conv2D: 64 filters, kernel=4, stride=1
ReLU
Fully Connected: 256
ReLU
Fully Connected: Number of actions (policy logits)


---

### Critic Network

Conv2D: identical convolutional backbone
Fully Connected: 256
ReLU
Fully Connected: 1 (state value)

