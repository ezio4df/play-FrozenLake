# FrozenLake RL Experiments

Playing around with DQN on FrozenLake....

- **`01_frozenlake_4x4.ipynb`**: ✅  
  4×4 grid, no slip (deterministic), fixed map. Solved easily with a simple MLP. Good sanity check.

  ![](files/frozenlake_4x4-episode-0.gif)

- **`02_frozenlake_8x8.ipynb`**: ✅  
  8×8, no slip (deterministic), fixed map. Also solved with MLP — stable ~0.8+ reward.

  ![](files/frozenlake_8x8-episode-0.gif)


- **`03_frozenlake_8x8_slippery.ipynb`**: ✅  
  8×8, slippery (stochastic), fixed map
  First try somehow gave ~0.6–0.7 avg reward… but I was accidentally feeding only the agent’s position (not the full
  map).  
  ![](files/frozenlake_8x8_slippery-episode-0.gif)

  When I switched to **visual input (84×84 rendered frames + coordinate channels)** to prepare for random maps,
  performance became unstable — best I got was **-0.07 avg reward at ep 20k**, with wild fluctuations. Still tuning.



- **`04_frozenlake_8x8_random.ipynb`**: ⏳  
  8×8, no slip (deterministic), random map
  **8×8, deterministic, new solvable random map every 2000 episodes.**  
  Agent sees the **full rendered screen + x/y coordinate channels** so it can (in theory) generalize.  
  Result: it **learns each map reliably**, but **resets to -1.3 every time the map changes**. Takes ~2000 episodes to
  recover and reach **+0.3–0.4 avg reward**. Not perfect, but it *does* adapt! Training videos show the whole journey —
  failures, breakthroughs, and all.

- **`05_frozenlake_8x8_random_slippery.ipynb`**: ⏳  
  8x8, slippery, random maps

