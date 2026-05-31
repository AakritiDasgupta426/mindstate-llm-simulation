# MINDSTATE

A research framework for studying belief propagation, consensus formation, and adversarial memory manipulation in networks of Large Language Model (LLM) agents.

## Overview

MINDSTATE investigates how beliefs evolve within multi-agent AI systems and how vulnerable those systems are to misinformation introduced through memory. The framework simulates networks of interacting LLM agents that maintain memory, exchange information, update beliefs over time, and respond to adversarial inputs.

As AI systems increasingly operate in collaborative environments, understanding how misinformation spreads between agents becomes an important safety challenge. MINDSTATE was developed to explore these risks and quantify the effects of memory manipulation on collective reasoning.

This work was presented at MIT URTC.

## Research Questions

MINDSTATE explores several key questions:

- How quickly do LLM agents reach consensus?
- How does belief propagation differ from classical consensus models?
- Can a single false memory influence an entire network?
- How resilient are multi-agent systems to adversarial manipulation?
- Do network structure and agent characteristics affect resistance to misinformation?

## System Architecture

The framework consists of:

- **Memory-Aware Agents** – agents that maintain evolving memories and beliefs over time
- **Multi-Agent Network Layer** – communication and interaction between agents
- **Belief Tracking System** – measurement of stance evolution and consensus formation
- **Adversarial Memory Injection Engine** – controlled insertion of false memories
- **Drift Detection Module** – analysis of belief divergence and manipulation effects
- **Visualization Tools** – generation of experimental plots and analysis

## Repository Structure

```text
basic_agent.py
memory_agent.py
multiagent_network.py
memory_manipulation_experiment.py
drift_detector.py
mindstate_visualizations.py
large_scale_test.py
run_simulation.py
```

### Core Components

| File | Purpose |
|--------|---------|
| `basic_agent.py` | Baseline agent implementation |
| `memory_agent.py` | Memory-aware agent behavior |
| `multiagent_network.py` | Network structure and agent interactions |
| `memory_manipulation_experiment.py` | Adversarial memory injection experiments |
| `drift_detector.py` | Belief drift and divergence analysis |
| `mindstate_visualizations.py` | Experimental visualization tools |
| `large_scale_test.py` | Large-scale simulation testing |
| `run_simulation.py` | Main simulation entry point |

## Methodology

1. Initialize a network of LLM agents.
2. Assign initial beliefs and memory states.
3. Allow agents to interact across multiple rounds.
4. Track belief evolution and consensus formation.
5. Inject false memories into selected agents.
6. Measure propagation, divergence, and manipulation success.
7. Compare outcomes against classical consensus baselines.

## Key Findings

Experimental results indicate:

- LLM-agent networks often reached consensus significantly faster than classical consensus models.
- False-memory injections were capable of influencing network-wide beliefs.
- Manipulation effects propagated beyond directly targeted agents.
- Network structure affected the speed and magnitude of belief propagation.
- Multi-agent systems demonstrated measurable vulnerability to adversarial memory attacks.

## Future Work

- Implement memory verification mechanisms
- Explore adversarial defense strategies
- Evaluate larger agent populations
- Investigate heterogeneous agent architectures
- Extend experiments to additional reasoning tasks and domains

## Author

**Aakriti Dasgupta**  
Electrical and Computer Engineering  
Purdue University

## Citation

If you use this work, please cite:

```text
Dasgupta, A. MINDSTATE: Systematic Analysis of Belief Manipulation
Vulnerabilities in Multi-Agent LLM Networks.
```
