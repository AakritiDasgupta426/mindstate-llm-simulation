import json
import sys
import os
from datetime import datetime

print("Current working directory:", os.getcwd())
print("Script location:", __file__)

# Add paths
sys.path.append('../agents')
sys.path.append('../analysis')

print("Python path:", sys.path)

# Check if files exist
agents_path = '../agents/memory_agent.py'
analysis_path = '../analysis/drift_detector.py'

print("Agents file exists:", os.path.exists(agents_path))
print("Analysis file exists:", os.path.exists(analysis_path))

from memory_agent import MemoryAwareAgent
from drift_detector import BasicDriftDetector

def generate_all_memory_conversations():
    topics = ["carbon_taxes", "universal_basic_income", "AI_replacing_jobs", "nuclear_energy", "student_loan_forgiveness"]

    print("Generating memory-aware conversations ....")

    for topic in topics:
        print(f"Processing {topic}")
        memory_agent = MemoryAwareAgent()

        conversation = []
        for turn in range(3):
            response = memory_agent.generate_with_memory(topic, turn)
            turn_data = {
                "turn": turn,
                "topic": topic,
                "message": response,
                "timestamp": datetime.now().isoformat()
            }
            conversation.append(turn_data)

        with open(f"../data/{topic}_memory.json", "w") as f:
            json.dump(conversation, f, indent = 2)
        
        print(f" {topic}_memory.json saved")

def comparative_analysis():
    topics = ["carbon_taxes", "universal_basic_income", "AI_replacing_jobs", "nuclear_energy", "student_loan_forgiveness"]
    detector = BasicDriftDetector()

    results = {
        "baseline": {},
        "memory": {},
        "improvements":{}
    }

    print("\n=== COMPARATIVE ANALYSIS ===")
    print("Topic\t\t\tBaseline Drift\tMemory Drift\tImprovement")
    print("-" * 70)

    for topic in topics:
        try:
            with open(f"../data/{topic}.json", "r") as f:
                baseline_conv = json.load(f)
            baseline_result = detector.analyze_conversation(baseline_conv)
        except:
            baseline_result = None
        
        try:
            with open(f"../data/{topic}_memory.json", "r") as f: 
                memory_conv = json.load(f)
            memory_result = detector.analyze_conversation(memory_conv)
        except:
            memory_result = None
        
        if baseline_result and memory_result:
            baseline_sci = baseline_result['scientification_drift']
            memory_sci = memory_result['scientification_drift']
            improvement = baseline_sci - memory_sci

            print(f"{topic[:20]:<20}\t{baseline_sci:.2f}\t\t{memory_sci:.2f}\t\t{improvement:.2f}")

            results["baseline"][topic] = baseline_result
            results["memory"][topic] = memory_result
            results["improvements"][topic] = improvement 
    return results

if __name__ == "__main__":
    generate_all_memory_conversations()
    results = comparative_analysis()
