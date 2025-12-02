"""
Main script to run the complete RFE with drift experiment

Usage: python run.py
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from rfe_drift.env import DriftGridWorld, DriftType, DriftSchedule
from rfe_drift.exploration import UCRLRFE
from rfe_drift.representations import FixedEncoder, DriftAwareEncoder, RepresentationTrainer
from rfe_drift.rl import DQNAgent
from rfe_drift.utils import MetricsTracker, GoalRewardFunction


# Configuration
CONFIG = {
    "grid_size": 10,
    "drift_type": DriftType.GOAL_SHIFT,
    "drift_strength": 0.7,
    "drift_schedule": DriftSchedule.SUDDEN,
    "drift_time": 200,
    "num_exploration_steps": 10000,
    "num_train_episodes": 300,
    "num_eval_episodes": 50,
    "seed": 42,
}

def main():
    Path("results").mkdir(exist_ok=True)
    
    print("="*60)
    print("RFE with Distributional Drift - Simplified")
    print("="*60)
    
    # Step 1: Reward-free exploration
    print("\n[1/4] Reward-Free Exploration...")
    env = DriftGridWorld(
        grid_size=CONFIG["grid_size"],
        drift_type=CONFIG["drift_type"],
        drift_strength=CONFIG["drift_strength"],
        drift_schedule=CONFIG["drift_schedule"],
        drift_time=CONFIG["drift_time"],
        seed=CONFIG["seed"],
    )
    
    explorer = UCRLRFE(state_dim=CONFIG["grid_size"]**2, action_dim=4)
    state, _ = env.reset()
    
    for step in range(CONFIG["num_exploration_steps"]):
        action = explorer.select_action(state)
        next_state, _, terminated, truncated, _ = env.step(action)
        explorer.update(state, action, next_state)
        
        if (step + 1) % 2000 == 0:
            coverage = explorer.get_state_coverage()
            print(f"  Step {step+1}: Coverage = {coverage:.3f}")
        
        state = next_state if not (terminated or truncated) else env.reset()[0]
    
    replay_buffer = explorer.get_replay_buffer()
    print(f"  Collected {len(replay_buffer)} transitions")
    
    # Step 2: Train representations
    print("\n[2/4] Training Representations...")
    
    fixed_encoder = FixedEncoder(input_dim=2, hidden_dim=64, output_dim=32)
    RepresentationTrainer(fixed_encoder).train_forward_dynamics(replay_buffer, num_epochs=20)
    
    drift_encoder = DriftAwareEncoder(input_dim=2, hidden_dim=64, output_dim=32)
    RepresentationTrainer(drift_encoder).train_forward_dynamics(replay_buffer, num_epochs=20)
    
    print("  Encoders trained")
    
    # Step 3: Train downstream RL agents
    print("\n[3/4] Training Downstream RL Agents...")
    
    results = {}
    
    for encoder, encoder_name in [(fixed_encoder, "Fixed"), (drift_encoder, "Drift-Aware")]:
        print(f"\n  Training {encoder_name} agent...")
        
        env = DriftGridWorld(
            grid_size=CONFIG["grid_size"],
            drift_type=CONFIG["drift_type"],
            drift_strength=CONFIG["drift_strength"],
            drift_schedule=CONFIG["drift_schedule"],
            drift_time=CONFIG["drift_time"],
            seed=CONFIG["seed"] + 10,
        )
        
        agent = DQNAgent(action_dim=4, encoder=encoder, epsilon=1.0, epsilon_decay=0.995)
        state, info = env.reset()
        
        episode_rewards = []
        episode_count = 0
        episode_reward = 0
        
        for step in range(CONFIG["num_train_episodes"] * 100):
            time = env.step_count / (CONFIG["drift_time"] * 2)
            action = agent.select_action(state, time=time, training=True)
            next_state, _, terminated, truncated, info = env.step(action)
            
            reward_fn = GoalRewardFunction(goals=info["goals"])
            reward = reward_fn(state, action, next_state)
            agent.update(state, action, reward, next_state, terminated, time=time)
            
            episode_reward += reward
            
            if terminated or truncated:
                episode_rewards.append(episode_reward)
                
                if (episode_count + 1) % 50 == 0:
                    avg = np.mean(episode_rewards[-50:])
                    print(f"    Episode {episode_count+1}: Avg reward = {avg:.3f}")
                
                state, info = env.reset()
                episode_reward = 0
                episode_count += 1
                
                if episode_count >= CONFIG["num_train_episodes"]:
                    break
            else:
                state = next_state
        
        results[encoder_name] = {'agent': agent, 'train_rewards': episode_rewards}
    
    # Step 4: Evaluate
    print("\n[4/4] Evaluation...")
    
    eval_results = {}
    
    for encoder_name in ["Fixed", "Drift-Aware"]:
        agent = results[encoder_name]['agent']
        
        env = DriftGridWorld(
            grid_size=CONFIG["grid_size"],
            drift_type=CONFIG["drift_type"],
            drift_strength=CONFIG["drift_strength"],
            drift_schedule=CONFIG["drift_schedule"],
            drift_time=CONFIG["drift_time"],
            seed=CONFIG["seed"] + 20,
        )
        
        agent.epsilon = 0.1
        
        # Evaluate before drift
        env.step_count = 0
        rewards_before = []
        for _ in range(CONFIG["num_eval_episodes"]//2):
            state, info = env.reset()
            env.step_count = 0
            reward_fn = GoalRewardFunction(goals=info["goals"])
            ep_reward = 0
            
            for _ in range(100):
                action = agent.select_action(state, time=0.0, training=False)
                next_state, _, terminated, truncated, _ = env.step(action)
                reward = reward_fn(state, action, next_state)
                ep_reward += reward
                if terminated or truncated or env.step_count >= CONFIG["drift_time"]:
                    break
                state = next_state
            
            rewards_before.append(ep_reward)
        
        # Evaluate after drift
        env.step_count = CONFIG["drift_time"] + 10
        env._apply_drift()
        rewards_after = []
        for _ in range(CONFIG["num_eval_episodes"]//2):
            state, info = env.reset()
            env.step_count = CONFIG["drift_time"] + 10
            env._apply_drift()
            reward_fn = GoalRewardFunction(goals=info["goals"])
            ep_reward = 0
            
            for _ in range(100):
                action = agent.select_action(state, time=1.0, training=False)
                next_state, _, terminated, truncated, _ = env.step(action)
                reward = reward_fn(state, action, next_state)
                ep_reward += reward
                if terminated or truncated:
                    break
                state = next_state
            
            rewards_after.append(ep_reward)
        
        eval_results[encoder_name] = {
            'before': rewards_before,
            'after': rewards_after,
            'mean_before': np.mean(rewards_before),
            'mean_after': np.mean(rewards_after),
        }
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    for encoder_name in ["Fixed", "Drift-Aware"]:
        r = eval_results[encoder_name]
        drop = r['mean_before'] - r['mean_after']
        rel_drop = (drop / r['mean_before'] * 100) if r['mean_before'] > 0 else 0
        
        print(f"\n{encoder_name}:")
        print(f"  Before drift: {r['mean_before']:.3f}")
        print(f"  After drift:  {r['mean_after']:.3f}")
        print(f"  Drop:         {drop:.3f} ({rel_drop:.1f}%)")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Training curves
    for encoder_name in ["Fixed", "Drift-Aware"]:
        rewards = results[encoder_name]['train_rewards']
        axes[0].plot(rewards, label=encoder_name, alpha=0.7)
    axes[0].axvline(CONFIG["drift_time"]//100, color='red', linestyle='--', label='Drift')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Training Performance')
    axes[0].legend()
    axes[0].grid(True)
    
    # Before/After comparison
    x = np.arange(2)
    width = 0.35
    axes[1].bar(x - width/2, [eval_results['Fixed']['mean_before'], eval_results['Fixed']['mean_after']], 
                width, label='Fixed', alpha=0.8)
    axes[1].bar(x + width/2, [eval_results['Drift-Aware']['mean_before'], eval_results['Drift-Aware']['mean_after']], 
                width, label='Drift-Aware', alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['Before Drift', 'After Drift'])
    axes[1].set_ylabel('Mean Reward')
    axes[1].set_title('Performance Comparison')
    axes[1].legend()
    axes[1].grid(True, axis='y')
    
    # Performance drop
    drops = [
        (eval_results['Fixed']['mean_before'] - eval_results['Fixed']['mean_after']) / eval_results['Fixed']['mean_before'] * 100 if eval_results['Fixed']['mean_before'] > 0 else 0,
        (eval_results['Drift-Aware']['mean_before'] - eval_results['Drift-Aware']['mean_after']) / eval_results['Drift-Aware']['mean_before'] * 100 if eval_results['Drift-Aware']['mean_before'] > 0 else 0,
    ]
    axes[2].bar(['Fixed', 'Drift-Aware'], drops, color=['red', 'orange'], alpha=0.7)
    axes[2].set_ylabel('Performance Drop (%)')
    axes[2].set_title('Robustness to Drift\n(Lower is Better)')
    axes[2].grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/final_results.png', dpi=150)
    print(f"\nResults saved to results/final_results.png")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()

