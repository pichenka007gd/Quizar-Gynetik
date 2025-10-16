# Quizar Gynetik

Quizar Gynetik is a lightweight, high-performance library for **fast agent-based simulations with real-time visualization** using **Pygame**.  

It allows you to quickly create agents, define their behavior, and watch them learn and evolve in a simulated environment. The library handles all the heavy lifting: **genetic algorithms, population management, and rendering**. You just need to inject the core logic of your agents and the world, and Quizar Gynetik takes care of the rest.  

---

## Features

- **Rapid prototyping**: define your agent and world logic, library handles the simulation loop.  
- **Real-time visualization**: built on Pygame for smooth rendering.  
- **Fast performance**: core calculations are implemented in C, so large populations run efficiently.  
- **Extensible**: easily customize agent inputs, step logic, rewards, and world interactions.  

---

## How it works

1. **Define your agent** by inheriting from `BaseAgent` and implementing:
   - `input()` — returns normalized inputs (sensors) for the agent.  
   - `step(out)` — processes NN outputs and returns reward.  

2. **Define your world** by inheriting from `World`:
   - Initialize global objects (e.g., targets, obstacles).  
   - Optionally override `step()` for custom simulation logic.  

3. The library handles:
   - Population evolution via **GeneticMLP / GAController**  
   - Agent updates and scoring  
   - Visualization and GUI controls  

---

## Quick Start Example

Clone the repository, build, and run a sample simulation:

```bash
git clone https://github.com/pichenka007gd/Quizar-Gynetik.git
cd Quizar-Gynetik
make
pip install -r requirements.txt
python3 sample_cursor.py
