# Usage Guide

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the experiment:**
   ```bash
   python run.py
   ```

3. **View results:**
   - Check `results/final_results.png` for visualization
   - Training curves, before/after drift comparison, performance drop

## What It Does

The `run.py` script executes the complete pipeline:

1. **Reward-Free Exploration** (10,000 steps)
   - UCRL-RFE algorithm explores the gridworld
   - Collects trajectories without rewards
   
2. **Representation Learning**
   - Trains Fixed encoder (ignores drift)
   - Trains Drift-Aware encoder (uses time information)
   
3. **Downstream RL Training** (300 episodes each)
   - Trains DQN agents using both encoders
   - Agents learn to reach goals with reward signals
   
4. **Evaluation** (50 episodes)
   - Tests performance before drift (steps 0-199)
   - Tests performance after drift (steps 200+)
   - Measures performance drop for each encoder

## Configuration

Edit `run.py` to change parameters:

```python
CONFIG = {
    "grid_size": 10,              # Grid size (10×10)
    "drift_strength": 0.7,         # How much goals move (0-1)
    "drift_time": 200,             # When drift occurs
    "num_exploration_steps": 10000,# RFE steps
    "num_train_episodes": 300,     # RL training episodes
    "num_eval_episodes": 50,       # Evaluation episodes
}
```

## Expected Runtime

- Total: ~10-15 minutes
- Exploration: ~1 minute
- Representation training: ~2 minutes
- Downstream training: ~5-10 minutes (2 encoders)
- Evaluation: ~1 minute

## Understanding Results

The key metric is **Performance Drop**:
- **Lower drop** = more robust to drift
- Drift-aware encoder should show smaller drop

Example good result:
```
Fixed:       Before 0.80 → After 0.40 (50% drop)
Drift-Aware: Before 0.75 → After 0.60 (20% drop)
```

This shows drift-aware maintains performance better under drift.

