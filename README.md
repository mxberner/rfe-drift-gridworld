# Robust Reward-Free Exploration under Distributional Drift

Most reinforcement learning methods rely on dense, well-shaped rewards, which are often unavailable, biased, or expensive to engineer. Reward-free exploration (RFE) instead learns good state representations or exploratory policies without using rewards, and only introduces rewards later for downstream tasks.
However, most existing RFE frameworks assume a static environment. In realistic settings, the data distribution can drift over time: goals move, dynamics change, or noise increases. This project explores how reward-free methods behave under such distributional drift.

---

## High Level Workflow

- Build a synthetic MDP (GridWorld) with tunable drift.
- Pretrain reward-free*representations using UCRL-RFE and baselines.
- Introduce downstream tasks with rewards after pretraining.
- Compare how quickly / robustly different representations adapt under drift.

---

## Goals

1. **Environment: Drift-Enabled GridWorld**

   Implement a small GridWorld-style MDP with:
   - Drift strength: how much the transition or reward structure changes.
   - Drift schedule: when drift happens (e.g., sudden jump, gradual shift, periodic).
   - Examples:
     - Shifting goal locations.
     - Changing transition noise.
     - Altering blocked cells or wall layouts.

2. **Reward-Free Exploration with UCRL-RFE**

   Apply a reward-free exploration algorithm (e.g. UCRL-RFE) to:
   - Collect trajectories without rewards.
   - Maximize state coverage.
   - Produce a replay buffer or dataset for later representation learning.

3. **Representation Pretraining**

   Train two families of state encoders:

   - Fixed-environment encoder 
     - Pretrained on data from a single (or early) environment configuration.
     - Ignores later drift during pretraining.
   - Drift-aware encoder**
     - Pretrained across time with drifting dynamics.
     - May condition on time, drift index, or inferred context.
     - Goal: learn stable or adaptable features under nonstationarity.

4. **Downstream Rewarded Tasks**

   After pretraining:
   - Introduce explicit reward functions
   - Train simple RL agents
   - Evaluate:
     - Learning speed
     - Final performance / Reward Earned
     - Representation stability

5. **Baselines and Comparisons**

   Metrics to track:
   - Coverage of state space over time.
   - Downstream sample efficiency.
   - Performance drop when drift occurs.
   - Representation similarity / drift across environments.
