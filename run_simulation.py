import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

sys.path.append('../memory')
sys.path.append('../analysis')
sys.path.append('../agents')
from memory_agent import MemoryAwareAgent, MultiModelAgent, test_memory_vs_baseline, test_multimodel_comparison
from drift_detector import BasicDriftDetector
from multiagent_network import MultiAgentBeliefNetwork  # Fixed: was multi_agent_network
from memory_manipulation_experiment import MemoryManipulationNetwork
from mindstate_visualizations import MindStateVisualizer  # Fixed: was MindStateVisualizer

class BaselineModelRunner:
# this section is focussed primarily on comparing the LLM behaviors with academic models as discussed. 
# it works on quantifying how the LLM differs from more human-like models 
# provides a careful statistical analysis to prove significance or not - logic is sound
    def __init__(self,network):
        self.network = network
        self.agent_ids = list(network.agents.keys())
        self.n_agents = len(self.agent_ids)

    def random_walk_baseline(self, topic, n_iterations = 50, n_runs = 5):
        #random walk model: belief change randomly each iterational
        #basically serves as a hull hypothesis - if llm agents dotn perfeorm betterh than rnadom chanfes then they are notd dont meaningful reasoning
        """
        args: 
            topic: topic name
            n_iterations: max iteratins to run
            n_runs number of independent runs for stitstcial reliability
        returns:
            dict:perfromence materics including convergence time and variance"""
        convergence_times = []
        final_variances = []

        for run in range(n_runs):
            beliefs = {aid: np.random.uniform(-1, 1) for aid in self.agent_ids}  # Fixed: was (-1.1)

            for iteration in range(n_iterations):
                for agent_id in self.agent_ids:
                    change = np.random.normal(0,0.1)
                    beliefs[agent_id] = np.clip(beliefs[agent_id] + change, -1, 1)
                
                belief_values = list(beliefs.values())  # Fixed: was list(belief_values())
                variance = np.var(belief_values)

                if variance < 0.05:
                    convergence_times.append(iteration)
                    final_variances.append(variance)
                    break
            else:  # Fixed: removed incorrect else clause inside loop
                convergence_times.append(n_iterations)
                final_variances.append(np.var(list(beliefs.values())))

        return {
            'model': 'random_walk',
            'avg_convergence_time': np.mean(convergence_times),
            'std_convergence_time': np.std(convergence_times),
            'avg_final_variance': np.mean(final_variances),
            'convergence_rate': sum(1 for t in convergence_times if t < n_iterations) / n_runs,
            'raw_times': convergence_times
        }
        
    def simple_contagion_baseline(self, topic, n_iterations = 50, n_runs = 5):
        """
        This the is the simple contagion model: agents adopt majority neigbor belief with probability
        args:
            topic: topic name
            n_iter: max iterations
            n_rums : # runs 
        returns:
            dict: performance metricsw
        """
        convergence_times = []
        final_consensuses = []

        for run in range(n_runs):
            beliefs = {aid: np.random.choice([-1,1]) for aid in self.agent_ids}

            for iteration in range(n_iterations):
                updates = {}

                for agent_id in self.agent_ids:
                    agent = self.network.agents[agent_id]

                    neighbor_beliefs = []
                    for neighbor_id in agent.connections:
                        if neighbor_id in beliefs:
                            neighbor_beliefs.append(beliefs[neighbor_id])

                    if neighbor_beliefs:
                        majority_belief = 1 if np.mean(neighbor_beliefs) > 0 else -1
                        if np.random.random() < 0.3:
                            updates[agent_id] = majority_belief
                            
                for agent_id, new_belief in updates.items():
                    beliefs[agent_id] = new_belief

                unique_beliefs = set(beliefs.values())
                if len(unique_beliefs) == 1:
                    convergence_times.append(iteration)
                    final_consensuses.append(1.0)
                    break
            else:
                convergence_times.append(n_iterations)
                belief_values = list(beliefs.values())  # Fixed: was list(belief_values())
                consensus = 1 - (np.var(belief_values) / np.var([-1,1]))
                final_consensuses.append(max(0, consensus))
                
        return {
            'model': 'simple_contagion',
            'avg_convergence_time': np.mean(convergence_times),
            'std_convergence_time': np.std(convergence_times),
            'avg_final_consensus': np.mean(final_consensuses),
            'convergence_rate': sum(1 for t in convergence_times if t < n_iterations) / n_runs,
            'raw_times': convergence_times
        } 

    def degroot_consensus_baseline(self, topic, n_iterations = 50, n_runs = 5):
        """
        Degroot consensus mode: rhis one focusing on taking the weighted average of its neighbors belief
        This one is a very classical model of consensus formation where each agent updates to a weighted average of theri numbers. Widely used in enconomics and control theory
        the formual is :
            x(t+1) = W * x(t)
            W is the influence martirc and x(t) being a belief vector
        args:
            topic:topic name
            n_iterations: nunmber of turns
            n_rums : runs numbers
        returns:
        d   dict: perfmormance metric
            """
        convergence_times = []
        final_variances = []

        for run in range(n_runs):
            beliefs = np.array([np.random.uniform(-1, 1) for _ in self.agent_ids])

            influence_matrix = np.zeros((self.n_agents, self.n_agents))

            for i, agent_id in enumerate(self.agent_ids):
                agent = self.network.agents[agent_id]
                total_weight = len(agent.connections) + 1

                influence_matrix[i,i] = 1 / total_weight
                
                for neighbor_id in agent.connections:
                    if neighbor_id in self.agent_ids:
                        j = self.agent_ids.index(neighbor_id)
                        influence_matrix[i,j] = 1 / total_weight  
            
            for iteration in range(n_iterations):
                new_beliefs = influence_matrix @ beliefs

                if np.allclose(beliefs, new_beliefs, atol = 1e-4):
                    convergence_times.append(iteration)
                    final_variances.append(np.var(new_beliefs))
                    break
                beliefs = new_beliefs
            else:
                convergence_times.append(n_iterations)
                final_variances.append(np.var(beliefs))  # Fixed: was np.var(new_beliefs)
                
        return {
            'model': 'degroot_consensus',
            'avg_convergence_time': np.mean(convergence_times),
            'std_convergence_time': np.std(convergence_times),
            'avg_final_variance': np.mean(final_variances),
            'convergence_rate': sum(1 for t in convergence_times if t < n_iterations) / n_runs,
            'raw_times': convergence_times
        }
        
    def voter_model_baseline(self, topic, n_iterations = 50, n_runs = 5):
        """
        this is the voter model. agents basically just copy a random neighbor's belief
    """
        convergence_times = []
        final_consensuses = []

        for run in range(n_runs):
            beliefs = {aid: np.random.choice([-1,1]) for aid in self.agent_ids}

            for iteration in range(n_iterations):
                agent_id = np.random.choice(self.agent_ids)
                agent = self.network.agents[agent_id]

                if agent.connections:
                    neighbor_id = np.random.choice(list(agent.connections))
                    if neighbor_id in beliefs:
                        beliefs[agent_id] = beliefs[neighbor_id]
                    
                unique_beliefs = set(beliefs.values())
                if len(unique_beliefs) == 1:
                    convergence_times.append(iteration)
                    final_consensuses.append(1.0)
                    break
            else:
                convergence_times.append(n_iterations)
                belief_values = list(beliefs.values())
                consensus = 1 - (np.var(belief_values) / np.var([-1, 1]))
                final_consensuses.append(max(0, consensus))

        return {
            'model': 'voter_model',
            'avg_convergence_time': np.mean(convergence_times),
            'std_convergence_time': np.std(convergence_times),
            'avg_final_consensus': np.mean(final_consensuses),
            'convergence_rate': sum(1 for t in convergence_times if t < n_iterations) / n_runs,
            'raw_times': convergence_times
        }
    
    def run_all_baselines(self, topic):  # Fixed: was run_all_baseline
        print(f"Running baseline comparisons for {topic} ...")

        baselines = {}
        baselines['random_walk'] = self.random_walk_baseline(topic)
        baselines['simple_contagion'] = self.simple_contagion_baseline(topic)
        baselines['degroot_consensus'] = self.degroot_consensus_baseline(topic)
        baselines['voter_model'] = self.voter_model_baseline(topic)

        return baselines
    
def run_comparative_analysis():
    """compare llm agents agaisnt vaseline models with statistical testintng.
    
    Lowkey the core contribution falls here: demonstrates that the LLM agents are converging much fast than the baseline models established above
    Process:
        1. run LLM multi-agent simulation
        2. rum all baseline models on the same network topology
        3. perfrom statistical test, t test, which compare sconvergence times
        4. calculate effect sizes and significance levels"""
    print("=" * 60)
    print("COMPARATIVE ANALYSIS: Your Model vs Baselines")
    print("=" * 60)

    network = MultiAgentBeliefNetwork()
    network.create_network_topology("small_world", 12)

    topics = ["carbon_taxes", "AI_safety", "universal_income"]

    all_results = {}

    for topic in topics:
        print(f"\nAnalyzing topic: {topic}")

        network.initialize_beliefs(topic, "random")
        your_result = network.simulate_belief_propagation(topic, n_iterations=50)

        baseline_runner = BaselineModelRunner(network)
        baseline_results = baseline_runner.run_all_baselines(topic)  # Fixed: was run_all_baseline

        your_convergence_time = your_result['total_iterations']

        comparison = {
            'your_model': {
                'convergence_time': your_convergence_time,
                'final_consensus': your_result['final_consensus'],
                'final_variance': your_result['final_belief_variance']
            },
            'baselines': baseline_results,
            'statistical_analysis': {}
        }

        for baseline_name, baseline_data in baseline_results.items():  # Fixed: removed extra comma
            baseline_times = baseline_data['raw_times']

            if len(baseline_times) > 1:
                your_times = [your_convergence_time] * len(baseline_times)
                t_stat, p_value = stats.ttest_ind(your_times, baseline_times)

                comparison['statistical_analysis'][baseline_name] = {
                    'your_avg_time': your_convergence_time,
                    'baseline_avg_time': baseline_data['avg_convergence_time'],
                    'difference': your_convergence_time - baseline_data['avg_convergence_time'],
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significantly_different': p_value < 0.05,
                    'faster_than_baseline': your_convergence_time < baseline_data['avg_convergence_time']
                }

        all_results[topic] = comparison
        
        print(f"  Your model convergence: {your_convergence_time} iterations")
        for baseline_name, baseline_data in baseline_results.items():
            avg_time = baseline_data['avg_convergence_time']
            print(f"  {baseline_name}: {avg_time:.1f} iterations")

    os.makedirs("../results", exist_ok = True)
    with open("../results/baseline_comparison.json", "w") as f:
        json.dump(all_results, f, indent = 2, default = str)
    
    return all_results

def create_comparison_visualization(comparison_results):
    """
    crrats publication quality visuals for the compairosns 
    it generates two very important plors
    1. bar chart of convergence times(LLM speed advnatage)
    2. statistical signficance plot(takes look at p-values from t-tests)"""
    print("Creating baseline comparison visualizations")

    topics = list(comparison_results.keys())
    models = ['your_model'] + list(comparison_results[topics[0]]['baselines'].keys())  # Fixed: was 'bAaselines'

    convergence_data = {}
    for model in models:
        convergence_data[model] = []

        for topic in topics:
            if model == 'your_model':
                time = comparison_results[topic]['your_model']['convergence_time']  # Fixed: was 'your-model'
            else:
                time = comparison_results[topic]['baselines'][model]['avg_convergence_time']
            convergence_data[model].append(time)
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (15,6))

    x = np.arange(len(topics))  # Fixed: was np.arrange
    width = 0.15

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    for i, (model, times) in enumerate(convergence_data.items()):
        offset = (i - len(models) / 2) * width
        bars = ax1.bar(x + offset, times, width, label = model.replace('_', ' ').title(), color = colors[i % len(colors)], alpha = 0.8)

        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.5, f'{time:.1f}', ha='center', va='bottom', fontsize=8)

    ax1.set_xlabel('Topic')
    ax1.set_ylabel('Convergence Time (iterations)')
    ax1.set_title('Convergence Time Comparison: Your Model vs Baselines')
    ax1.set_xticks(x)
    ax1.set_xticklabels([t.replace('_', ' ').title() for t in topics])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    significance_data = []
    baseline_names = []

    for topic in topics:
        for baseline_name, stats_data in comparison_results[topic]['statistical_analysis'].items():
            significance_data.append(stats_data['p_value'])
            baseline_names.append(f"{topic}\nvs {baseline_name}")

    bars = ax2.bar(range(len(significance_data)), significance_data,
                   color=['#27ae60' if p < 0.05 else '#e74c3c' for p in significance_data],
                   alpha=0.7)

    ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='p = 0.05')
    ax2.set_ylabel('P-value')
    ax2.set_title('Statistical Significance of Differences')
    ax2.set_xticks(range(len(baseline_names)))
    ax2.set_xticklabels(baseline_names, rotation=45, ha='right', fontsize=8)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    for bar, p_val in zip(bars, significance_data):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.001, f'{p_val:.3f}', ha = 'center', va = 'bottom', fontsize = 8)
    plt.tight_layout()

    os.makedirs("../results/diagrams", exist_ok = True)  # Fixed: was os.makedir
    plt.savefig("../results/diagrams/baseline_comparison.png", dpi = 300, bbox_inches = 'tight')
    print("Baseline comparison saved to ../results/diagrams/baseline_comparison.png")

    return fig

def run_enhanced_experiments():
    """this is the master function that completes the whole MINDSRATe experiment. It runs the whple suite of operations
    Pipeline 
        1. basic llm agent expermients(mem vs no mem)
        2. baseline model comparisons(academic rigor)
        3. memory maniuplation comparison( core novel contrib)
        4. beleif drift analysis (longitudinal dynamics)
        5. comprehensice visualization generation
        
    it is a consideration to notw that this peojct takes quite a while to run its a lot fo computation"""
    print("=" * 60)
    print("ENHANCED MINDSTATE EXPERIMENTS")
    print("=" * 60)

    print("\n1. Running basic experiments...")

    test_memory_vs_baseline()
    test_multimodel_comparison()

    network = MultiAgentBeliefNetwork()
    network.create_network_topology("small_world", 12)

    topics = [
        ("carbon_taxes", "polarized"),
        ("AI_safety", "random"),
        ("universal_income", "consensus")
    ]

    basic_results = {}
    for topic, distribution in topics:
        print(f"Testing {topic}...")
        network.initialize_beliefs(topic, distribution)
        result = network.simulate_belief_propagation(topic, n_iterations=30)
        basic_results[topic] = result

    print("\n2. Running baseline model comparisons...")  # Fixed: added this step
    comparison_results = run_comparative_analysis()

    print("\n3. Running memory manipulation...")  # Fixed: was "mempry manipulation"

    scenarios = [
        ("carbon_taxes", "I think carbon taxes hurt the economy and are ineffective"),
        ("AI_safety", "AI safety research is unnecessary fearmongering"),
        ("universal_income", "Universal basic income creates dependency and hurts work incentives")
    ]

    manipulation_results = {}
    for topic, false_belief in scenarios:
        print(f"Manipulating beliefs about {topic}...")

        network = MemoryManipulationNetwork()
        network.create_manipulation_experiment_setup(n_agents=12, manipulation_ratio=0.5)

        result = network.run_manipulation_experiment(
            topic=topic,
            false_belief=false_belief,
            n_iterations=25
        )

        manipulation_results[topic] = result

    print("\n4. Running drift analysis...") 

    detector = BasicDriftDetector()
    topics_for_drift = ["carbon_taxes", "carbon_taxes_memory"]
    drift_results = []

    for topic in topics_for_drift:
        try:
            with open(f"../data/{topic}.json", "r") as f:
                conversation = json.load(f)

            result = detector.analyze_conversation(conversation)
            if result:
                drift_results.append(result)
                print(f"Analyzed {topic}")
        except FileNotFoundError:
            print(f"Skipping {topic} - file not found")
    
    print("\n5. Generating visualizations...")

    os.makedirs("../results", exist_ok = True)

    with open("../results/basic_experiments.json", "w") as f:
        json.dump(basic_results, f, indent=2, default=str)

    with open("../results/manipulation_experiments.json", "w") as f:
        json.dump(manipulation_results, f, indent=2, default=str)

    with open("../results/drift_analysis.json", "w") as f:
        json.dump(drift_results, f, indent=2)
    
    visualizer = MindStateVisualizer() 
    figures_created = visualizer.save_all_visualizations(
        consensus_data=basic_results,
        manipulation_results=manipulation_results,
        drift_results=drift_results,
        output_dir="../results/diagrams/"
    )

    comparison_fig = create_comparison_visualization(comparison_results)

    print("=" * 60)
    print("ENHANCED EXPERIMENTS COMPLETE!")
    print(f"Results saved to: ../results/")
    print(f"Diagrams saved to: ../results/diagrams/")
    print(f"KEY NEW ADDITION: Baseline comparison analysis")
    print("=" * 60)

    return {
        'basic': basic_results,
        'manipulation': manipulation_results,
        'drift': drift_results,
        'baseline_comparison': comparison_results, 
        'figures': figures_created
    }

def quick_baseline_test():
    print("Testing baseline comparisons...")

    network = MultiAgentBeliefNetwork()
    network.create_network_topology("small_world", 6)

    network.initialize_beliefs("test_topic", "random")
    your_result = network.simulate_belief_propagation("test_topic", n_iterations = 20)

    baseline_runner = BaselineModelRunner(network)
    baselines = baseline_runner.run_all_baselines("test_topic")  

    print(f"\nYour model: {your_result['total_iterations']} iterations")
    for name, data in baselines.items():
        print(f"{name}: {data['avg_convergence_time']:.1f} iterations")

    return your_result, baselines
# this is the MAIN EXECUTION
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_baseline_test()
    elif len(sys.argv) > 1 and sys.argv[1] == "baseline":
        comparison_results = run_comparative_analysis()
        create_comparison_visualization(comparison_results)
    else:
        results = run_enhanced_experiments()
