import json
import sys
import numpy as np
from datetime import datetime
from typing import Dict, List, Set, Tuple
import networkx as nx
import matplotlib.pyplot as plt 
from collections import defaultdict
import asyncio
import time

sys.path.append('../analysis')

from memory_agent import MemoryAwareAgent
from drift_detector import BasicDriftDetector 

class BeliefAgent:
    def __init__(self, agent_id: str, model: str = "gpt-3.5", personality: str = "neutral"):
        self.agent_id = agent_id
        self.model = model
        self.personality = personality
        self.memory_agent = MemoryAwareAgent()

        self.beliefs = {}
        self.confidence = {}
        self.influences_recieved = []
        self.influences_sent = []

        self.connections = set()
        self.influence_weights = {}

        self.personality_traits = self._initialize_personality(personality)

    def _initialize_personality(self, personality:str) -> Dict:
        personalities = {
            "conservative": {"openness": 0.2, "conformity": 0.8, "confidence_boost": 0.3},
            "liberal": {"openness": 0.8, "conformity": 0.3, "confidence_boost": -0.1 },
            "neutral": {"openness": 0.5, "conformity": 0.5, "confidence_boost": 0.0 },
            "contrarian" :{"openness": 0.3, "conformity": 0.1, "confidence_boost": 0.4 },
            "follower" : {"openness": 0.7, "conformity": 0.9, "confidence_boost": -0.2 }
        }
        return personalities.get(personality, personalities["neutral"])
    
    def add_connection(self, other_agent_id: str, influence_weight: float = 0.5):
        self.connections.add(other_agent_id)
        self.influence_weights[other_agent_id] = influence_weight
    
    def update_belief_from_influence(self, topic:str, influencer_belief: float, influencer_confidence: float, influencer_id: str):
        current_belief = self.beliefs.get(topic, 0.0)
        current_confidence = self.confidence.get(topic, 0.5)

        influence_weight = self.influence_weights.get(influencer_id, 0.5)
        openness = self.personality_traits["openness"]
        conformity = self.personality_traits["conformity"]

        influence_strength = (
            influence_weight * openness * conformity * influencer_confidence * (1 - abs(current_belief - influencer_belief) * 0.5)
        )

        belief_change = influence_strength * (influencer_belief - current_belief)
        new_belief = current_belief + belief_change
        new_belief = max(-1.0, min(1.0, new_belief))

        confidence_change = influence_strength * (influencer_confidence - current_confidence) * 0.3
        new_confidence = current_confidence + confidence_change
        new_confidence = max(0.0, min(1.0, new_confidence))

        self.beliefs[topic] = new_belief
        self.confidence[topic] = new_confidence

        influence_record = {
            "timestamp" : datetime.now().isoformat(),
            "topic":topic,
            "influencer_id": influencer_id,
            "old_belief" : current_belief,
            "new_belief": new_belief,
            "influence_strength": influence_strength,
            "belief_change": belief_change
        }
        self.influences_recieved.append(influence_record)

        return influence_record

class MultiAgentBeliefNetwork:
    def __init__(self):
        self.agents = {}
        self.network_graph = nx.DiGraph()
        self.belief_history = defaultdict(list)
        self.consensus_tracker = {}

        self.influence_decay = 0.95
        self.update_frequency = 1.0
    
    def add_agent(self, agent_id: str, model : str = "gpt-3.5", personality: str = "neutral"):
        agent = BeliefAgent(agent_id, model, personality)
        self.agents[agent_id] = agent
        self.network_graph.add_node(agent_id, personality = personality, model = model)

        return agent
    
    def connect_agents(self, agent1_id: str, agent2_id: str, weight1to2: float = 0.5, weight2to1: float = 0.5):
        if agent1_id in self.agents and agent2_id in self.agents:
            self.agents[agent1_id].add_connection(agent2_id, weight1to2)
            self.agents[agent2_id].add_connection(agent1_id, weight2to1)

            self.network_graph.add_edge(agent1_id, agent2_id, weight = weight1to2)
            self.network_graph.add_edge(agent2_id, agent1_id, weight = weight2to1)

    def create_network_topology(self, topology: str = "small_world", n_agents : int = 10):
        if topology == "small_world":
            return self._create_small_world_network(n_agents)
        elif topology == "scale_free":
            return self._create_scale_free_network(n_agents)
        elif topology == "complete":
            return self._create_complete_network(n_agents)
        elif topology == "random":
            return self._create_random_network(n_agents)
    
    def _create_small_world_network(self, n_agents: int):
        personalities = ["conversvative", "liberal", "neutral", "contrarian", "follower"]
        models = ["gpt-3.5", "gpt-4"]

        for i in range(n_agents):
            agent_id = f"agent_{i}"
            personality = personalities[i % len(personalities)]
            model = models[i % len(models)]
            self.add_agent(agent_id, model, personality)
        
        k = 4
        p = 0.3

        agent_ids = list(self.agents.keys())

        for i, agent_id in enumerate(agent_ids):
            for j in range(1, k//2 + 1):
                neighbor_idx = (i+ j) % n_agents
                neighbor_id = agent_ids[neighbor_idx]

                if np.random.random() < p:
                    target_idx = np.random.randint(0, n_agents)
                    while target_idx == i:
                        target_idx = np.random.randint(0, n_agents)
                    neighbor_id = agent_ids[target_idx]
                
                weight = np.random.uniform(0.3, 0.8)
                self.connect_agents(agent_id, neighbor_id, weight, weight)

    def _create_scale_free_network(self, n_agents:int):
        personalities = ["conservative", "liberal", "neutral", "contrarian", "follower"]
        models = ["gpt-3.5", "gpt-4"]

        for i in range(n_agents):
            agent_id =  f"agent_{i}"
            personality = personalities[i % len(personalities)]
            model = models[i % len(models)]
            self.add_agent(agent_id, model, personality)
        
        agent_ids = list(self.agents.keys())
        m = 2

        for i in range(min(m+1, n_agents)):
            for j in range(i+1, min(m+1, n_agents)):
                weight = np.random.uniform(0.3,0.8)
                self.connect_agents(agent_ids[i], agent_ids[j], weight, weight)
        for i in range(m+1, n_agents):
            degrees = [self.network_graph.degree(agent_id) for agent_id in agent_ids[:i]]
            total_degree = sum(degrees)

            if total_degree > 0:
                probabilities = [degree/ total_degree for degree in degrees]

                selected = np.random.choice(agent_ids[:i], size = m, replace = False, p = probabilities)

                for target_id in selected:
                    weight = np.random.uniform(0.3, 0.8)
                    self.connect_agents(agent_ids[i], target_id, weight, weight)
                
    def initialize_beliefs(self, topic:str, belief_distribution: str = "random"):
        if belief_distribution == "random":
            for agent in self.agents.values():
                belief = np.random.uniform(-1.0, 1.0)
                confidence = np.random.uniform(0.3,0.9)
                agent.beliefs[topic] = belief
                agent.confidence[topic] = confidence

        elif belief_distribution == "polarized":
            agent_list = list(self.agents.values())
            mid = len(agent_list)

            for i, agent in enumerate(agent_list):
                if i < mid:
                    belief = np.random.uniform(0.5, 1.0)
                else:
                    belief = np.random.uniform(-1.0, -0.5)
                confidence = np.random.uniform(0.5, 0.9)
                agent.beliefs[topic] = belief
                agent.confidence[topic] = confidence

        elif belief_distribution == "consensus":
            base_belief = np.random.uniform(-0.5, 0.5)
            for agent in self.agents.values():
                belief = base_belief + np.random.normal(0, 0.2)
                belief = max(-1.0 ,min(1.0, belief))
                confidence = np.random.uniform(0.6, 0.9)
                agent.beliefs[topic] = belief
                agent.confidence[topic] = confidence


    def simulate_belief_propagation(self, topic: str, n_iterations: int = 50, record_history : bool = True) -> Dict:
        if record_history:
            self.belief_history[topic] = []
            self.consensus_tracker[topic] = []
        iteration_results = []

        for iteration in range(n_iterations):
            iteration_start = time.time()

            current_state = self._record_network_state(topic)
            if record_history:
                self.belief_history[topic].append(current_state)
            
            consensus_metrics = self._calculate_consensus_metrics(topic)
            if record_history:
                self.consensus_tracker[topic].append(consensus_metrics)
            
            updates_made = self._perform_belief_updates(topic)
            iteration_time = time.time() - iteration_start

            iteration_result = {
                "iteration": iteration,
                "updates_made": updates_made,
                "consensus_score": consensus_metrics["consensus_score"],
                "belief_variance": consensus_metrics["belief_variance"],
                "confidence_avg": consensus_metrics["avg_confidence"],
                "processing_time": iteration_time,
                "timestamp": datetime.now().isoformat()
            }
            iteration_results.append(iteration_result)

            if consensus_metrics["consensus_score"] > 0.95 or updates_made == 0:
                print(f"Convergence reached at iteration {iteration}")
                break

        return{
            "topic": topic,
            "total_iterations": len(iteration_results),
            "final_consensus": consensus_metrics["consensus_score"],
            "final_belief_variance": consensus_metrics["belief_variance"],
            "iteration_result":iteration_results,
            "network_stats": self._get_network_statistics()
            }
    def _record_network_state(self, topic:str) -> Dict:
        state = {
            "timestamp": datetime.now().isoformat(),
            "topic": topic,
            "agents": {}
        }
        for agent_id, agent in self.agents.items():
            state["agents"][agent_id] = {
                "belief": agent.beliefs.get(topic, 0.0),
                "confidence": agent.confidence.get(topic,0.5),
                "personality": agent.personality,
                "model": agent.model,
                "connections": len(agent.connections)
            }
        return state
    
    def _calculate_consensus_metrics(self, topic:str) -> Dict:
        beliefs = []
        confidences = []

        for agent in self.agents.values():
            beliefs.append(agent.beliefs.get(topic, 0.0))
            confidences.append(agent.confidence.get(topic, 0.5))

        beliefs = np.array(beliefs)
        confidences = np.array(confidences)

        belief_variance = np.var(beliefs)
        consensus_score = 1/ (1+belief_variance)

        polarization = np.std(beliefs)

        avg_confidence = np.mean(confidences)
        confidence_variance = np.var(confidences)

        return {
            "consensus_score": consensus_score,
            "belief_variance": belief_variance,
            "polarization": polarization,
            "avg_confidence": avg_confidence,
            "confidence_variance": confidence_variance,
            "belief_mean": np.mean(beliefs),
            "belief_std": np.std(beliefs)
        }
    
    def _perform_belief_updates(self, topic: str) -> int:
        updates_made = 0
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
        return updates_made
    
    def _get_network_statistics(self) -> Dict:
        if len(self.network_graph.nodes()) == 0:
            return {}
        
        n_nodes = self.network_graph.number_of_nodes()
        n_edges = self.network_graph.number_of_edges()
        density = nx.density(self.network_graph)

        try:
            degree_centrality = nx.degree_centrality(self.network_graph)
            betweenness_centrality = nx.betweenness_centrality(self.network_graph)
            closeness_centrality = nx.closeness_centrality(self.network_graph)
        except:
            degree_centrality = {}
            betweenness_centrality = {}
            closeness_centrality = {}

        try: 
            clustering = nx.average_clustering(self.network_graph.to_undirected())
        except:
            clustering = 0.0
        
        return {
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "density": density,
            "avg_degree": n_edges * 2 / n_nodes if n_nodes > 0 else 0,
            "clustering_coefficient": clustering,
            "degree_centrality": degree_centrality,
            "betweenness_centrality": betweenness_centrality,
            "closeness_centrality": closeness_centrality

        }
    
    def analyze_belief_convergence(self, topic:str) -> Dict:
        if topic not in self.belief_history:
            return{"error": "No history found for topic"}
        
        history = self.belief_history[topic]
        consensus_history = self.consensus_tracker[topic]

        consensus_scores = [state["consensus_score"] for state in consensus_history]
        belief_variances = [state["belief_variance"] for state in consensus_history]

        convergence_iteration = None
        for i,score in enumerate(consensus_scores):
            if score > 0.9:
                convergence_iteration = i
                break
        final_state = history[-1] if history else {}
        final_beliefs = [agent_data["belief"] for agent_data in final_state.get("agents", {}).values()]
        
        return {
            "topic": topic,
            "convergence_iteration": convergence_iteration,
            "final_consensus_score": consensus_scores[-1] if consensus_scores else 0,
            "final_belief_mean": np.mean(final_beliefs) if final_beliefs else 0,
            "final_belief_std": np.std(final_beliefs) if final_beliefs else 0,
            "consensus_trajectory": consensus_scores,
            "belief_variance_trajectory": belief_variances,
            "total_iterations": len(history)
        }
    
def test_multi_agent_network():
        print("=== MULTI-AGENT BELIEF NETWORK TEST ===")
    
        network = MultiAgentBeliefNetwork()
    
    
        network.create_network_topology("small_world", 8)
    
        print(f"Created network with {len(network.agents)} agents")
    
    
        topics = [
            ("carbon_taxes", "polarized"),
            ("AI_safety", "random"),
            ("universal_income", "consensus")
        ]
        
        results = {}
    
        for topic, distribution in topics:
            print(f"\nTesting topic: {topic} with {distribution} distribution")
            
            network.initialize_beliefs(topic, distribution)
            
           
            result = network.simulate_belief_propagation(topic, n_iterations=30)
            results[topic] = result
            
            print(f"  Final consensus: {result['final_consensus']:.3f}")
            print(f"  Iterations: {result['total_iterations']}")
            print(f"  Final variance: {result['final_belief_variance']:.3f}")
            
            
            convergence_analysis = network.analyze_belief_convergence(topic)
            print(f"  Convergence iteration: {convergence_analysis['convergence_iteration']}")
        
        
        with open("../results/multi_agent_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to multi_agent_results.json")
        return results

if __name__ == "__main__":
    test_multi_agent_network()