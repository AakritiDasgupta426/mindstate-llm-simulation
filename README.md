# MINDSTATE

Multi-agent LLM simulation system for studying belief propagation and manipulation under adversarial memory injection.

## Overview

Modern AI systems are evolving from single agents into networks of collaborating agents that share memory, exchange information, and converge on decisions.

MINDSTATE is a research-driven simulation platform I built to study how false beliefs spread through multi-agent LLM systems, how quickly these systems reach consensus, and how vulnerable they are to adversarial memory injection.

This project was developed as part of my research and presented at MIT URTC.

## What It Does

- Simulates memory-aware LLM agents interacting in a network
- Models belief updates over repeated interaction rounds
- Injects false memory into a selected agent and tracks propagation
- Measures consensus speed, divergence, and manipulation success
- Visualizes belief trajectories and network-wide belief shifts

## System Architecture

MINDSTATE is organized into four layers:

1. **Agent Layer**  
   Memory-aware agents maintain internal belief states and update them over time.

2. **Network Layer**  
   Agents are connected through configurable topologies that control how beliefs spread.

3. **Experiment Layer**  
   Experiments evaluate consensus formation, divergence, and vulnerability to manipulation.

4. **Visualization Layer**  
   Outputs include interpretable plots of belief trajectories, convergence, and takeover dynamics.

## Key Findings

- LLM agents reached consensus significantly faster than classical social dynamics baselines
- A single injected false memory was able to spread across the full network in repeated trials
- Manipulation succeeded even across varied personalities and network structures
- Topics with weaker prior reinforcement were more vulnerable to false-memory takeover

## Next Steps

- Add a minimal runnable simulation
- Add sample experiment configurations
- Add result visualizations
- Prototype defenses such as memory authentication and adversarial verification

## Author

Aakriti Dasgupta  
Electrical and Computer Engineering, Purdue University
