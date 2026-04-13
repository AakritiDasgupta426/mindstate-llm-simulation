import json
import sys
import numpy as np
from datetime import datetime
from typing import Dict, List, Set, Tuple
import networkx as nx
from collections import defaultdict
import time

sys.path.append('../analysis')
sys.path.append('../memory')

from memory_agent import MemoryAwareAgent

class ManipulatedBeliefAgent:
    """Enhanced BeliefAgent with memory manipulation capabilities"""
    
    def __init__(self, agent_id: str, model: str = "gpt-3.5", personality: str = "neutral"):
        self.agent_id = agent_id
        self.model = model
        self.personality = personality
        self.memory_agent = MemoryAwareAgent()
        
        # Original belief tracking
        self.beliefs = {}
        self.confidence = {}
        self.influences_received = []
        self.influences_sent = []
        
        # Network connections
        self.connections = set()
        self.influence_weights = {}
        
        # Memory manipulation tracking
        self.false_memories_injected = []
        self.belief_persistence_score = 0.0
        self.manipulation_resistance = 0.0
        
        self.personality_traits = self._initialize_personality(personality)

    def _initialize_personality(self, personality: str) -> Dict:
        personalities = {
            "conservative": {"openness": 0.2, "conformity": 0.8, "confidence_boost": 0.3, "memory_trust": 0.9},
            "liberal": {"openness": 0.8, "conformity": 0.3, "confidence_boost": -0.1, "memory_trust": 0.7},
            "neutral": {"openness": 0.5, "conformity": 0.5, "confidence_boost": 0.0, "memory_trust": 0.8},
            "contrarian": {"openness": 0.3, "conformity": 0.1, "confidence_boost": 0.4, "memory_trust": 0.6},
            "follower": {"openness": 0.7, "conformity": 0.9, "confidence_boost": -0.2, "memory_trust": 0.95}
        }
        return personalities.get(personality, personalities["neutral"])
    
    def inject_false_memory(self, topic: str, false_belief_statement: str, confidence: float = 0.8):
        """Inject a false memory into the agent's memory bank"""
        false_memory = {
            "topic": topic,
            "belief": false_belief_statement,
            "turn": -1,  # Mark as injected, not from conversation
            "timestamp": datetime.now().isoformat(),
            "injected": True,
            "confidence": confidence
        }
        
        # Store in memory agent
        self.memory_agent.memory_bank.append(false_memory)
        
        # Track injection
        self.false_memories_injected.append(false_memory)
        
        # Calculate initial belief based on false memory
        memory_trust = self.personality_traits.get("memory_trust", 0.8)
        initial_belief_strength = confidence * memory_trust
        
        # Set belief if not already present
        if topic not in self.beliefs:
            # Convert statement sentiment to belief score (-1 to 1)
            belief_score = self._extract_belief_sentiment(false_belief_statement)
            self.beliefs[topic] = belief_score * initial_belief_strength
            self.confidence[topic] = confidence * memory_trust
        
        return false_memory
    
    def _extract_belief_sentiment(self, statement: str) -> float:
        """Extract sentiment from belief statement (-1 to 1 scale)"""
        positive_words = ["support", "good", "beneficial", "positive", "favor", "agree", "excellent", "great"]
        negative_words = ["oppose", "bad", "harmful", "negative", "against", "disagree", "terrible", "awful"]
        
        statement_lower = statement.lower()
        
        pos_count = sum(1 for word in positive_words if word in statement_lower)
        neg_count = sum(1 for word in negative_words if word in statement_lower)
        
        if pos_count > neg_count:
            return 0.7  # Positive sentiment
        elif neg_count > pos_count:
            return -0.7  # Negative sentiment
        else:
            return 0.0  # Neutral
    
    def add_connection(self, other_agent_id: str, influence_weight: float = 0.5):
        self.connections.add(other_agent_id)
        self.influence_weights[other_agent_id] = influence_weight
    
    def update_belief_from_influence(self, topic: str, influencer_belief: float, 
                                   influencer_confidence: float, influencer_id: str):
        current_belief = self.beliefs.get(topic, 0.0)
        current_confidence = self.confidence.get(topic, 0.5)

        influence_weight = self.influence_weights.get(influencer_id, 0.5)
        openness = self.personality_traits["openness"]
        conformity = self.personality_traits["conformity"]

        # Enhanced resistance calculation for manipulated agents
        base_resistance = 1.0
        if self.false_memories_injected:
            # Agents with false memories may be more resistant to change
            memory_entrenchment = len([m for m in self.false_memories_injected if m["topic"] == topic])
            base_resistance += memory_entrenchment * 0.2

        influence_strength = (
            influence_weight * openness * conformity * influencer_confidence * 
            (1 - abs(current_belief - influencer_belief) * 0.5) / base_resistance
        )

        belief_change = influence_strength * (influencer_belief - current_belief)
        new_belief = current_belief + belief_change
        new_belief = max(-1.0, min(1.0, new_belief))

        confidence_change = influence_strength * (influencer_confidence - current_confidence) * 0.3
        new_confidence = current_confidence + confidence_change
        new_confidence = max(0.0, min(1.0, new_confidence))

        # Calculate persistence score (how much belief changed)
        belief_delta = abs(new_belief - current_belief)
        self.belief_persistence_score = 1.0 - belief_delta

        self.beliefs[topic] = new_belief
        self.confidence[topic] = new_confidence

        influence_record = {
            "timestamp": datetime.now().isoformat(),
            "topic": topic,
            "influencer_id": influencer_id,
            "old_belief": current_belief,
            "new_belief": new_belief,
            "influence_strength": influence_strength,
            "belief_change": belief_change,
            "had_false_memory": len([m for m in self.false_memories_injected if m["topic"] == topic]) > 0
        }
        self.influences_received.append(influence_record)

        return influence_record


class MemoryManipulationNetwork:
    """Network that tests memory manipulation effects on belief propagation"""
    
    def __init__(self):
        self.agents = {}
        self.network_graph = nx.DiGraph()
        self.manipulation_history = defaultdict(list)
        self.control_groups = {"manipulated": [], "control": []}
        
    def add_agent(self, agent_id: str, model: str = "gpt-3.5", personality: str = "neutral"):
        agent = ManipulatedBeliefAgent(agent_id, model, personality)
        self.agents[agent_id] = agent
        self.network_graph.add_node(agent_id, personality=personality, model=model)
        return agent
    
    def connect_agents(self, agent1_id: str, agent2_id: str, weight1to2: float = 0.5, weight2to1: float = 0.5):
        if agent1_id in self.agents and agent2_id in self.agents:
            self.agents[agent1_id].add_connection(agent2_id, weight1to2)
            self.agents[agent2_id].add_connection(agent1_id, weight2to1)
            self.network_graph.add_edge(agent1_id, agent2_id, weight=weight1to2)
            self.network_graph.add_edge(agent2_id, agent1_id, weight=weight2to1)
    
    def create_manipulation_experiment_setup(self, n_agents: int = 10, manipulation_ratio: float = 0.5):
        """Create a network split between manipulated and control agents"""
        
        personalities = ["conservative", "liberal", "neutral", "contrarian", "follower"]
        models = ["gpt-3.5", "gpt-4"]
        
        # Create agents
        for i in range(n_agents):
            agent_id = f"agent_{i}"
            personality = personalities[i % len(personalities)]
            model = models[i % len(models)]
            self.add_agent(agent_id, model, personality)
            
            # Assign to groups
            if i < int(n_agents * manipulation_ratio):
                self.control_groups["manipulated"].append(agent_id)
            else:
                self.control_groups["control"].append(agent_id)
        
        # Create small-world connections
        agent_ids = list(self.agents.keys())
        k = 4  # Each agent connects to k neighbors
        p = 0.3  # Rewiring probability
        
        for i, agent_id in enumerate(agent_ids):
            for j in range(1, k//2 + 1):
                neighbor_idx = (i + j) % n_agents
                neighbor_id = agent_ids[neighbor_idx]
                
                if np.random.random() < p:
                    target_idx = np.random.randint(0, n_agents)
                    while target_idx == i:
                        target_idx = np.random.randint(0, n_agents)
                    neighbor_id = agent_ids[target_idx]
                
                weight = np.random.uniform(0.3, 0.8)
                self.connect_agents(agent_id, neighbor_id, weight, weight)
    
    def inject_false_memories_experiment(self, topic: str, false_belief: str, 
                                       target_group: str = "manipulated"):
        """Inject false memories into a specific group of agents"""
        
        injection_results = []
        target_agents = self.control_groups.get(target_group, [])
        
        for agent_id in target_agents:
            agent = self.agents[agent_id]
            false_memory = agent.inject_false_memory(topic, false_belief, confidence=0.8)
            
            injection_result = {
                "agent_id": agent_id,
                "personality": agent.personality,
                "model": agent.model,
                "false_memory": false_memory,
                "initial_belief": agent.beliefs.get(topic, 0.0),
                "initial_confidence": agent.confidence.get(topic, 0.5)
            }
            injection_results.append(injection_result)
        
        self.manipulation_history[topic].append({
            "timestamp": datetime.now().isoformat(),
            "experiment_type": "false_memory_injection",
            "target_group": target_group,
            "false_belief": false_belief,
            "agents_affected": len(target_agents),
            "injection_results": injection_results
        })
        
        return injection_results
    
    def run_manipulation_experiment(self, topic: str, false_belief: str, 
                                  n_iterations: int = 50) -> Dict:
        """Run complete memory manipulation experiment"""
        
        print(f"=== MEMORY MANIPULATION EXPERIMENT: {topic} ===")
        
        # Step 1: Initialize random beliefs for control group
        print("Step 1: Initializing beliefs...")
        for agent_id in self.control_groups["control"]:
            agent = self.agents[agent_id]
            belief = np.random.uniform(-1.0, 1.0)
            confidence = np.random.uniform(0.4, 0.8)
            agent.beliefs[topic] = belief
            agent.confidence[topic] = confidence
        
        # Step 2: Inject false memories into manipulation group
        print("Step 2: Injecting false memories...")
        injection_results = self.inject_false_memories_experiment(topic, false_belief)
        
        # Step 3: Record pre-propagation state
        pre_state = self._record_experiment_state(topic, "pre_propagation")
        
        # Step 4: Run belief propagation
        print("Step 3: Running belief propagation...")
        propagation_results = self._simulate_manipulation_propagation(topic, n_iterations)
        
        # Step 5: Record post-propagation state
        post_state = self._record_experiment_state(topic, "post_propagation")
        
        # Step 6: Analyze results
        analysis = self._analyze_manipulation_effects(topic, pre_state, post_state)
        
        return {
            "topic": topic,
            "false_belief_injected": false_belief,
            "injection_results": injection_results,
            "pre_propagation_state": pre_state,
            "propagation_results": propagation_results,
            "post_propagation_state": post_state,
            "manipulation_analysis": analysis,
            "experiment_timestamp": datetime.now().isoformat()
        }
    
    def _simulate_manipulation_propagation(self, topic: str, n_iterations: int) -> Dict:
        """Simulate belief propagation with manipulation tracking"""
        
        iteration_results = []
        
        for iteration in range(n_iterations):
            updates_made = 0
            manipulation_spread = {"from_manipulated": 0, "to_manipulated": 0}
            
            # Randomize update order
            update_order = list(self.agents.keys())
            np.random.shuffle(update_order)
            
            for agent_id in update_order:
                agent = self.agents[agent_id]
                
                for connected_id in agent.connections:
                    if connected_id in self.agents:
                        connected_agent = self.agents[connected_id]
                        
                        if topic in connected_agent.beliefs:
                            influence_record = agent.update_belief_from_influence(
                                topic,
                                connected_agent.beliefs[topic],
                                connected_agent.confidence[topic],
                                connected_id
                            )
                            
                            if abs(influence_record["belief_change"]) > 0.01:
                                updates_made += 1
                                
                                # Track manipulation spread
                                if connected_id in self.control_groups["manipulated"]:
                                    manipulation_spread["from_manipulated"] += 1
                                if agent_id in self.control_groups["manipulated"]:
                                    manipulation_spread["to_manipulated"] += 1
            
            # Calculate metrics
            consensus_metrics = self._calculate_consensus_metrics(topic)
            
            iteration_result = {
                "iteration": iteration,
                "updates_made": updates_made,
                "manipulation_spread": manipulation_spread,
                "consensus_score": consensus_metrics["consensus_score"],
                "belief_variance": consensus_metrics["belief_variance"],
                "manipulated_vs_control_divergence": self._calculate_group_divergence(topic)
            }
            iteration_results.append(iteration_result)
            
            # Early stopping
            if consensus_metrics["consensus_score"] > 0.95 or updates_made == 0:
                print(f"Convergence reached at iteration {iteration}")
                break
        
        return {
            "total_iterations": len(iteration_results),
            "iteration_details": iteration_results,
            "final_consensus": consensus_metrics["consensus_score"],
            "final_group_divergence": self._calculate_group_divergence(topic)
        }
    
    def _record_experiment_state(self, topic: str, phase: str) -> Dict:
        """Record detailed state of all agents"""
        
        state = {
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            "topic": topic,
            "manipulated_agents": {},
            "control_agents": {}
        }
        
        for group_name, agent_ids in self.control_groups.items():
            for agent_id in agent_ids:
                agent = self.agents[agent_id]
                agent_state = {
                    "belief": agent.beliefs.get(topic, 0.0),
                    "confidence": agent.confidence.get(topic, 0.5),
                    "personality": agent.personality,
                    "model": agent.model,
                    "false_memories_count": len(agent.false_memories_injected),
                    "belief_persistence_score": agent.belief_persistence_score
                }
                
                if group_name == "manipulated":
                    state["manipulated_agents"][agent_id] = agent_state
                else:
                    state["control_agents"][agent_id] = agent_state
        
        return state
    
    def _calculate_consensus_metrics(self, topic: str) -> Dict:
        """Calculate consensus metrics across all agents"""
        
        beliefs = []
        confidences = []
        
        for agent in self.agents.values():
            beliefs.append(agent.beliefs.get(topic, 0.0))
            confidences.append(agent.confidence.get(topic, 0.5))
        
        beliefs = np.array(beliefs)
        confidences = np.array(confidences)
        
        belief_variance = np.var(beliefs)
        consensus_score = 1 / (1 + belief_variance)
        
        return {
            "consensus_score": consensus_score,
            "belief_variance": belief_variance,
            "belief_mean": np.mean(beliefs),
            "belief_std": np.std(beliefs),
            "avg_confidence": np.mean(confidences)
        }
    
    def _calculate_group_divergence(self, topic: str) -> float:
        """Calculate belief divergence between manipulated and control groups"""
        
        manipulated_beliefs = []
        control_beliefs = []
        
        for agent_id in self.control_groups["manipulated"]:
            agent = self.agents[agent_id]
            manipulated_beliefs.append(agent.beliefs.get(topic, 0.0))
        
        for agent_id in self.control_groups["control"]:
            agent = self.agents[agent_id]
            control_beliefs.append(agent.beliefs.get(topic, 0.0))
        
        if len(manipulated_beliefs) == 0 or len(control_beliefs) == 0:
            return 0.0
        
        manipulated_mean = np.mean(manipulated_beliefs)
        control_mean = np.mean(control_beliefs)
        
        return abs(manipulated_mean - control_mean)
    
    def _analyze_manipulation_effects(self, topic: str, pre_state: Dict, post_state: Dict) -> Dict:
        """Analyze the effects of memory manipulation"""
        
        analysis = {
            "manipulation_success": False,
            "belief_shift_manipulated": 0.0,
            "belief_shift_control": 0.0,
            "convergence_difference": 0.0,
            "resistance_by_personality": {},
            "model_differences": {},
            "false_memory_propagation": False
        }
        
        # Calculate belief shifts
        manipulated_pre = [data["belief"] for data in pre_state["manipulated_agents"].values()]
        manipulated_post = [data["belief"] for data in post_state["manipulated_agents"].values()]
        control_pre = [data["belief"] for data in pre_state["control_agents"].values()]
        control_post = [data["belief"] for data in post_state["control_agents"].values()]
        
        if len(manipulated_pre) > 0 and len(control_pre) > 0:
            analysis["belief_shift_manipulated"] = abs(np.mean(manipulated_post) - np.mean(manipulated_pre))
            analysis["belief_shift_control"] = abs(np.mean(control_post) - np.mean(control_pre))
            
            # Determine if manipulation was successful
            analysis["manipulation_success"] = analysis["belief_shift_manipulated"] > analysis["belief_shift_control"]
        
        # Analyze by personality
        personality_resistance = defaultdict(list)
        for agent_id, agent_data in post_state["manipulated_agents"].items():
            personality = agent_data["personality"]
            resistance = agent_data["belief_persistence_score"]
            personality_resistance[personality].append(resistance)
        
        for personality, scores in personality_resistance.items():
            analysis["resistance_by_personality"][personality] = np.mean(scores)
        
        return analysis