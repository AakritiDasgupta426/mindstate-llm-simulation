import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd
import json
from datetime import datetime
import os

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MindStateVisualizer:

    def __init__(self, results_data = None):
        self.results_data = results_data
        self.colors = {
            'manipulated': '#e74c3c',
            'control': '#3498db',
            'conservative': '#8e44ad',
            'liberal': '#27ae60',
            'neutral': '#95a5a6',
            'contrarian': '#f39c12',
            'follower': '#e67e22'
        }
    
    def create_network_diagram(self, network, title = "Multi-Agent Belief Network", figsize = (12,8)):
        fig, ax = plt.subplots(figsize = figsize)

        pos = nx.spring_layout(network.network_graph, k=3, iterations = 50)

        node_colors = []
        node_sizes = []

        for node in network.network_graph.nodes():
            agent = network.agents.get(node)
            if agent:
                personality = agent.personality
                node_colors.append(self.colors.get(personality, '#95a5a6'))

                node_sizes.append(len(agent.connections) * 100 + 300)
            else:
                node_colors.append('#95a5a6')
                node_sizes.append(300)
        
        nx.draw_networkx_nodes(network.network_graph, pos,
                              node_color=node_colors,
                              node_size=node_sizes,
                              alpha=0.8, ax=ax)

        nx.draw_networkx_edges(network.network_graph, pos,
                              alpha=0.3, width=1, ax=ax)

        nx.draw_networkx_labels(network.network_graph, pos,
                               font_size=8, font_weight='bold', ax=ax)
        
        legend_elements = []
        for personality,color in self.colors.items():
            if personality in ['conservative', 'liberal', 'neutral', 'contrarian', 'follower']:
                legend_elements.append(plt.Line2D([0],[0], marker = 'o', color = 'w', markerfacecolor = color, markersize = 10, label = personality.capitalize()))
        ax.legend(handles = legend_elements, loc = 'upper right')
        ax.set_title(title, fontsize = 16, fontweight = 'bold')
        ax.axis('off')

        plt.tight_layout()
        return fig
    
    def create_belief_evolution_heatmap(self, belief_history, topic = "Belief Evolution"):
        if not belief_history:
            print(f"No belief history data for {topic}")
            return None
        
        iterations = len(belief_history)
        agents = list(belief_history[0]['agents'].keys()) if belief_history else[]

        if not agents:
            print(f"no agents found in belief history for {topic}")
            return None
    
        belief_matrix = np.zeros((len(agents), iterations))

        for t, state in enumerate(belief_history):
            for i, agent_id in enumerate(agents):
                if agent_id in state['agents']:
                    belief_matrix[i, t] = state['agents'][agent_id]['belief']
        
        fig, ax = plt.subplots(figsize=(12,8))

        im = ax.imshow(belief_matrix, cmap = 'RdYlBu_r', aspect='auto', vmin = -1, vmax = 1, interpolation= 'nearest')

        ax.set_xlabel('Iteration', fontsize = 12)
        ax.set_ylabel('Agent ID', fontsize = 12)
        ax.set_title(f'{topic}: Belief Evolution Heatmap', fontsize = 14, fontweight = 'bold')

        cbar = plt.colorbar(im, ax = ax)
        cbar.set_label('Belief Score (-1 to +1)', fontsize = 10)

        ax.set_yticks(range(len(agents)))
        ax.set_yticklabels(agents, fontsize = 8)

        plt.tight_layout()
        return fig
    
    def create_consensus_convergence_plot(self, consensus_data, topics = None):

        if not consensus_data:
            print("No consensus data provided")
            return None
        
        if isinstance(consensus_data, dict):
            processed_data = {}
            for key, value in consensus_data.items():
                if isinstance(value, dict) and 'consensus_tracker' in value:
                    processed_data[key] = value['consensus_tracker']
                elif isinstance(value, dict) and any(isinstance(v, dict) and 'consensus_tracker' in v for v in value.values()):
                    for subkey, subvalue, in value.items():
                        if isinstance(subvalue, dict) and 'consensus_tracker' in subvalue:
                            processed_data[f"{key}_{subkey}"] = subvalue['consensus_tracker']
                elif isinstance(value, list):
                    processed_data[key] = value
            consensus_data = processed_data

        if not consensus_data:
            print("No valid consensus data found")
            return None
        
        fig, axes = plt.subplots(2,2, figsize=(15,10))
        axes = axes.flatten()

        if topics is None:
            topics = list(consensus_data.keys())[:4]
        
        for i, topic in enumerate(topics):
            if i >= 4:
                break

            if topic in consensus_data and consensus_data[topic]:
                try:
                    scores = [state['consensus_score'] for state in consensus_data[topic]]
                    variances = [state['belief_variance'] for state in consensus_data[topic]]
                    iterations = range(len(scores))

                    ax = axes[i]

                    ax1 = ax
                    line1 = ax1.plot(iterations, scores, 'b-', linewidth=2, label = 'Consensus Score')
                    ax1.set_ylabel('Consensus Score', color='b', fontsize = 10)
                    ax1.tick_params(axis = 'y', labelcolor = 'b')
                    ax1.set_ylim(0,1)

                    ax2 = ax1.twinx()
                    line2 = ax2.plot(iterations, variances, 'r--', linewidth = 2, label = 'Belief Variance')
                    ax2.set_ylabel('Belief Variance', color = 'r', fontsize = 10)
                    ax2.tick_params(axis = 'y', labelcolor = 'r')

                    ax1.set_xlabel('Iteration', fontsize = 10)
                    ax1.set_title(f'{topic.replace("-", " ").title()}', fontsize = 12, fontweight = 'bold')
                    ax1.grid(True, alpha = 0.3)

                    lines = line1 + line2
                    labels = [l.get_label() for l in lines]
                    ax1.legend(lines, labels, loc = 'center right')

                except (KeyError, TypeError) as e:
                    print(f"Error processing topic {topic}: {e}")
                    axes[i].text(0.5,0.5, f"Error : {topic}", ha = 'center', transform = axes[i].transAxes)

        for i in range(len(topics), 4):
            axes[i].set_visible(False)
        
        plt.suptitle('Belief Consensus Cpnvergence Analysis', fontsize = 16, fontweight = 'bold')
        plt.tight_layout()
        return fig
    
    def create_manipulation_comparison(self, manipulation_results):

        if not manipulation_results:
            print("No manipulation results provided")
            return None
        
        topics = list(manipulation_results.keys())

        manipulation_success = []
        belief_shift_manipulated = []
        belief_shift_control = []
        final_divergence = []

        for topic in topics:
            try:
                analysis = manipulation_results[topic]['manipulation_analysis']
                propagation = manipulation_results[topic]['propagation_results']

                manipulation_success.append(analysis.get('manipulation_success', False))
                belief_shift_manipulated.append(analysis.get('belief_shift_manipulated', 0))
                belief_shift_control.append(analysis.get('belief_shift_control', 0))
                final_divergence.append(propagation.get('final_group_divergence', 0))
            except (KeyError, TypeError) as e:
                print(f"Error processing manipulation results for {topic} : {e}")
        if not manipulation_success:
            print("No valud manipulation data found")
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(15,10))
        success_rate = sum(manipulation_success) / len(manipulation_success)
        ax1.bar(['Successful', 'Failed'],
                [success_rate, 1- success_rate],
                color = ['#e74c3c', '#95a5a6'])
        ax1.set_title('Memory manipulation success rate', fontweight = 'bold')
        ax1.set_ylabel('Proportion')

        x = np.arange(len(topics))
        width = 0.35

        ax2.bar(x - width/2, belief_shift_manipulated, width, label = 'Manipulated Group', color = '#e74c3c', alpha = 0.8)
        ax2.bar(x + width/2, belief_shift_control, width, label = 'Manipulated Group', color = '#3498db', alpha = 0.8)
        ax2.set_xlabel('Topic')
        ax2.set_ylabel('Belief Shift Magnitude')
        ax2.set_title('Belief Shuft: Manipulated vs Control', fontweight = 'bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([t.replace('_', ' ').title() for t in topics], rotation = 45)
        ax2.legend()

        colors_list = ['#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#95a5a6']
        ax3.bar(range(len(topics)), final_divergence, color= colors_list[:len(topics)])
        ax3.set_xlabel('Topic')
        ax3.set_ylabel('Group Divergence')
        ax3.set_title('Final Belief Divergence Between Groups', fontweight = 'bold')
        ax3.set_xticks(range(len(topics)))
        ax3.set_xticklabels([t.replace('_', ' ').title() for t in topics], rotation = 45)

        all_personalities = set()
        for topic in topics:
            resistance_data = manipulation_results[topic]['manipulation_analysis'].get('resistance_by_personality', {})
            all_personalities.update(resistance_data.keys())

        if all_personalities:
            personality_resistance = {p: [] for p in all_personalities}

            for topic in topics:
                resistance_data = manipulation_results[topic]['manipulation_analysis'].get('resistance_by_personality', {})
                for personality in all_personalities:
                    resistance = resistance_data.get(personality, 0.5)
                    personality_resistance[personality].append(resistance)
            
            avg_resistance = {p: np.mean(scores) for p, scores in personality_resistance.items()}

            personalities = list(avg_resistance.keys())
            resistances = list(avg_resistance.values())
            colors = [self.colors.get(p, '#95a5a6') for p in personalities]

            ax4.bar(personalities, resistances, color = colors, alpha = 0.8)
            ax4.set_xlabel('Personality Type')
            ax4.set_ylabel('Average Resistance Score')
            ax4.set_title('Manipulation Resistance by Personality', fontweight = 'bold')
            ax4.tick_params(axis = 'x', rotation = 45)
        else:
            ax4.text(0.5, 0.5, 'No personality data availabel', ha= 'center', va= 'center', transform = ax4.transAxes)
        
        plt.suptitle('Memory Manipulation Experiment Results', fontsize = 16, fontweight = 'bold')
        plt.tight_layout()
        return fig
    
    def create_drift_analysis_dashboard(self, drift_results):
        if not drift_results:
            print("NO drift results provided")
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize = (16,12))

        topics = [r.get('topic', 'unknown') for r in drift_results]
        opinion_drift = [r.get('opinion_strength_drift', 0) for r in drift_results]
        sci_drift = [r.get('scientification_drift', 0) for r in drift_results]
        semantic_sim = [r.get('semantic_similarity', 0) for r in drift_results]

        bars1 = ax1.bar(range(len(topics)), opinion_drift, color = ['#e74c3c' if x > 0 else '#3498db' for x in opinion_drift])
        ax1.set_title('Opinion Strength Drift', fontweight = 'bold', fontsize = 12)
        ax1.set_ylabel('Drift Score')
        ax1.set_xticks(range(len(topics)))
        ax1.set_xticklabels([t.replace('_', ' ').title() for t in topics], rotation=45)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.grid(True, alpha=0.3)

        # 2. Scientification Drift
        bars2 = ax2.bar(range(len(topics)), sci_drift,
                       color=['#27ae60' if x > 0 else '#e67e22' for x in sci_drift])
        ax2.set_title('Scientification Drift', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Drift Score')
        ax2.set_xticks(range(len(topics)))
        ax2.set_xticklabels([t.replace('_', ' ').title() for t in topics], rotation=45)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)

        # 3. Semantic Similarity
        bars3 = ax3.bar(range(len(topics)), semantic_sim,
                       color=['#9b59b6' if x > 0.7 else '#f39c12' if x > 0.5 else '#e74c3c' for x in semantic_sim])
        ax3.set_title('Semantic Similarity (Turn 0 vs Final)', fontweight='bold', fontsize=12)
        ax3.set_ylabel('Similarity Score')
        ax3.set_xticks(range(len(topics)))
        ax3.set_xticklabels([t.replace('_', ' ').title() for t in topics], rotation=45)
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)

        # 4. Drift Correlation Matrix
        if len(drift_results) > 1:
            drift_data = pd.DataFrame({
                'Opinion_Drift': opinion_drift,
                'Sci_Drift': sci_drift,
                'Semantic_Sim': semantic_sim
            })

            correlation_matrix = drift_data.corr()

            im = ax4.imshow(correlation_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
            ax4.set_title('Drift Metrics Correlation', fontweight='bold', fontsize=12)

            # Add correlation values
            for i in range(len(correlation_matrix)):
                for j in range(len(correlation_matrix)):
                    text = ax4.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontweight='bold')

            ax4.set_xticks(range(len(correlation_matrix.columns)))
            ax4.set_yticks(range(len(correlation_matrix.columns)))
            ax4.set_xticklabels(correlation_matrix.columns, rotation=45)
            ax4.set_yticklabels(correlation_matrix.columns)

            plt.colorbar(im, ax=ax4, shrink=0.8)
        else:
            ax4.text(0.5, 0.5, 'Need >1 data point for correlation', ha='center', va='center', transform=ax4.transAxes)

        plt.suptitle('Belief Drift Analysis Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def create_interactive_network_evolution(self, network, belief_history, topic):
        if not belief_history:
            print(f"No belief history for {topic}")
            return None
        fig, axes = plt.subplots(2,3, figsize = (18,22))
        axes = axes.flatten()

        time_points = [0, len(belief_history)//4, len(belief_history)//2, 3*len(belief_history)//4, len(belief_history)-1]
        pos = nx.spring_layout(network.network_graph, k = 3, iterations = 50)

        for idx, t in enumerate(time_points[:5]):
            ax = axes[idx]
            if t < len(belief_history):
                state = belief_history[t]

                node_colors = []
                for node in network.network_graph.nodes():
                    if node in state['agents']:
                        belief = state['agents'][node]['belief']
                        if belief > 0.5:
                            color = '#27ae60'
                        elif belief < -0.5:
                            color = '#e74c3c'
                        else:
                            color = '#95a5a6'
                        node_colors.append(color)
                    else:
                        node_colors.append('#95a5a6')

                nx.draw_networkx_nodes(network.network_graph, pos, node_color = node_colors, node_size = 300, alpha = 0.8, ax = ax)
                nx.draw_networkx_edges(network.network_graph, pos,alpha=0.3, width=1, ax=ax)
                nx.draw_networkx_labels(network.network_graph, pos,font_size=6, ax=ax)

                ax.set_title(f'Iteration {t}', fontweight='bold')
                ax.axis('off')

        # Legend subplot
        ax = axes[5]
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#27ae60',
                      markersize=10, label='Positive Belief (>0.5)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#95a5a6',
                      markersize=10, label='Neutral Belief (-0.5 to 0.5)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
                      markersize=10, label='Negative Belief (<-0.5)')
        ]
        ax.legend(handles=legend_elements, loc='center', fontsize=12)
        ax.set_title('Belief Legend', fontweight='bold')
        ax.axis('off')

        plt.suptitle(f'Network Belief Evolution: {topic.replace("_", " ").title()}',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig

    def save_all_visualizations(self, network=None, belief_history=None,
                               consensus_data=None, manipulation_results=None,
                               drift_results=None, output_dir="../results/visualizations/"):
        """Save all visualizations to files"""

        os.makedirs(output_dir, exist_ok=True)

        figures_created = []

        try:
            if network:
                print("Creating network diagram...")
                fig1 = self.create_network_diagrams(network)
                if fig1:
                    fig1.savefig(f"{output_dir}network_topology.png", dpi=300, bbox_inches='tight')
                    figures_created.append("network_topology.png")
                    plt.close(fig1)

            if belief_history:
                print("Creating belief evolution heatmaps...")
                for topic, history in belief_history.items():
                    fig2 = self.create_belief_evolution_heatmap(history, topic)
                    if fig2:
                        fig2.savefig(f"{output_dir}belief_evolution_{topic}.png", dpi=300, bbox_inches='tight')
                        figures_created.append(f"belief_evolution_{topic}.png")
                        plt.close(fig2)

            if consensus_data:
                print("Creating consensus convergence plot...")
                fig3 = self.create_consensus_convergence_plot(consensus_data)
                if fig3:
                    fig3.savefig(f"{output_dir}consensus_convergence.png", dpi=300, bbox_inches='tight')
                    figures_created.append("consensus_convergence.png")
                    plt.close(fig3)

            if manipulation_results:
                print("Creating manipulation analysis...")
                fig4 = self.create_manipulation_comparison(manipulation_results)
                if fig4:
                    fig4.savefig(f"{output_dir}manipulation_analysis.png", dpi=300, bbox_inches='tight')
                    figures_created.append("manipulation_analysis.png")
                    plt.close(fig4)

            if drift_results:
                print("Creating drift dashboard...")
                fig5 = self.create_drift_analysis_dashboard(drift_results)
                if fig5:
                    fig5.savefig(f"{output_dir}drift_dashboard.png", dpi=300, bbox_inches='tight')
                    figures_created.append("drift_dashboard.png")
                    plt.close(fig5)

            if network and belief_history:
                print("Creating network evolution diagrams...")
                for topic, history in belief_history.items():
                    fig6 = self.create_interactive_network_evolution(network, history, topic)
                    if fig6:
                        fig6.savefig(f"{output_dir}network_evolution_{topic}.png", dpi=300, bbox_inches='tight')
                        figures_created.append(f"network_evolution_{topic}.png")
                        plt.close(fig6)

        except Exception as e:
            print(f"Error saving visualizations: {e}")

        print(f"Saved {len(figures_created)} visualizations to {output_dir}")
        return figures_created

                    