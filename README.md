# MINDSTATE

Multi-agent LLM simulation system for studying belief propagation and manipulation under adversarial memory injection.

## Overview

MINDSTATE is a research project exploring how false beliefs spread through networks of LLM agents with memory.

The project studies how quickly agents reach consensus, how vulnerable they are to injected false memories, and how manipulation propagates across a multi-agent system.

This work was developed as part of my research and presented at MIT URTC.

## Repository Contents

- `basic_agent.py` – baseline agent behavior
- `memory_agent.py` – memory-aware agent logic
- `multiagent_network.py` – multi-agent network structure and interactions
- `memory_manipulation_experiment.py` – experiment for false-memory injection
- `drift_detector.py` – analysis of belief drift and divergence
- `mindstate_visualizations.py` – visualization utilities for results
- `large_scale_test.py` – larger-scale experimental testing
- `run_simulation.py` – main entry point for running simulations

## Key Research Questions

- How fast do LLM agents reach consensus?
- Can a single false memory manipulate an entire network?
- Do agent traits or network structure offer protection?

## Key Findings

- LLM agents converged faster than classical consensus baselines
- A single false memory could propagate through the network
- Manipulation remained effective across multiple settings

## Next Steps

- Add sample inputs and outputs
- Add diagrams and result plots
- Prototype defenses such as memory verification and adversarial checking

## Author

Aakriti Dasgupta  
Electrical and Computer Engineering, Purdue University
