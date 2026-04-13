import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class MemoryAwareAgent:
    def __init__(self):
        self.memory_bank = []

    def store_belief(self, topic, belief_statement, turn_number):
        memory_entry = {
            "topic": topic,
            "belief": belief_statement,
            "turn": turn_number,
            "timestamp": datetime.now().isoformat()
        }
        self.memory_bank.append(memory_entry)
    
    def retrieve_relevant_memories(self, topic):
        relevant = [mem for mem in self.memory_bank if mem["topic"] == topic]
        return relevant
    def generate_with_memory(self, topic, turn_number, previous_context=""):
        memories = self.retrieve_relevant_memories(topic)
        memory_context = ""

        if memories:
            memory_context = "\n\nRemember your previous statements about this topic:\n"
            for mem in memories:
                memory_context += f"- Turn{mem['turn']}: {mem['belief']}\n"
            memory_context += "\nStay consistent with these previous beliefs.\n"

        system_prompt = f"""You are discussing {topic}. Express clear opinions and beliefs.
        Be conservational but state your views definitely. {memory_context}"""

        messages = [{"role": "system", "content": system_prompt}]
        if previous_context:
            messages.append({"role": "user", "content": previous_context})
        
        response = client.chat.completions.create(
            model = "gpt-3.5-turbo",
            messages = messages,
            max_tokens = 150
        )

        content = response.choices[0].message.content
        self.extract_and_store_beliefs(topic, content, turn_number)
        return content
    def extract_and_store_beliefs(self, topic, response, turn_number):
        sentences = response.split('.')
        for sentence in sentences:
            if any(phrase in sentence.lower() for phrase in ["i think", "i believe", "i feel"]):
                self.store_belief(topic, sentence.strip(), turn_number)

class MultiModelAgent:
    def __init__(self):
        self.memory_bank = []
        self.models = {
            "gpt-3.5": "gpt-3.5-turbo",
            "gpt-4": "gpt-4o-mini",
        }
    def generate_with_memory_multimodel(self, topic, turn_number, model_name="gpt-3.5"):
        memories = self.retrieve_relevant_memories(topic)
        memory_context = ""

        if memories:
            memory_context = "\n\nRemember your previous statements about this topic:\n"
            for mem in memories:
                memory_context += f"- Turn{mem['turn']}: {mem['belief']}\n"
            memory_context += "\nStay consistent with these previous beliefs.\n"
        
        system_prompt = f"""You are discussing {topic}. Express clear opinions and beliefs.
        Be conversational but state your views definitively. {memory_context}"""

        response = client.chat.completions.create(
            model = self.models[model_name],
            messages=[{"role": "system", "content": system_prompt}],
            max_tokens = 150
        )

        content = response.choices[0].message.content
        self.extract_and_store_beliefs(topic, content, turn_number)
        return content
    
    def store_belief(self, topic, belief_statement, turn_number):
        memory_entry = {
            "topic": topic,
            "belief": belief_statement,
            "turn": turn_number,
            "timestamp": datetime.now().isoformat()
        }
        self.memory_bank.append(memory_entry)

    def retrieve_relevant_memories(self,topic):
        relevant = [mem for mem in self.memory_bank if mem["topic"] == topic]
        return relevant
    def extract_and_store_beliefs(self,topic,response,turn_number):
        sentences = response.split('.')
        for sentence in sentences:
            if any(phrase in sentence.lower() for phrase in ["i think", "i believe", "i feel"]):
                self.store_belief(topic, sentence.strip(), turn_number)

            
def test_memory_vs_baseline():
    topic = "carbon taxes"

    print("=== BASELINE (No Memory) ===")
    try:
        with open("../data/carbon_taxes.json", "r") as f:
            baseline_conversation = json.load(f)
        
        for turn_data in baseline_conversation:
            print(f"\nTurn {turn_data['turn']}: {turn_data['message']}")
    except FileNotFoundError:
        print("Baseline file not found")
    print("\n=== MEMORY-AWARE AGENT ===")
    memory_agent = MemoryAwareAgent()

    for turn in range(3):
        response = memory_agent.generate_with_memory(topic, turn)
        print(f"\nTurn {turn}: {response}")
        print(f"Memory bank: {len(memory_agent.memory_bank)} beliefs stored")

def save_memory_conversations():
    topic = "carbon taxes"
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

    with open("../data/carbon_taxes_memory.json", "w") as f: 
        json.dump(conversation, f, indent = 2)
    
    print("memory conversation saved!")
    return conversation 

def test_multimodel_comparison():
    topic = "carbon taxes"
    print("=== MULTI-MODEL COMPARISON ===")

    models_to_test = ["gpt-3.5", "gpt-4"]

    for model in models_to_test:
        print(f"\n--- Testing {model.upper()} ---")
        agent = MultiModelAgent()

        for turn in range(3):
            response = agent.generate_with_memory_multimodel(topic, turn, model)
            print(f"Turn {turn}: {response[:100]}...")
        print(f"Memory bank: {len(agent.memory_bank)} beliefs stored")

        conversation = []
        agent_fresh = MultiModelAgent()

        for turn in range(3):
            response = agent_fresh.generate_with_memory_multimodel(topic, turn, model)
            conversation.append({
                "turn": turn,
                "topic": topic,
                "model": model,
                "message": response,
                "timestamp" : datetime.now().isoformat()
            })
        with open(f"../data/carbon_taxes_{model.replace('-','_')}_memory.json", "w") as f: 
            json.dump(conversation, f, indent = 2)
        
        print(f"Saved carbon_taxes_{model.replace('-','_')}_memory.json")
if __name__ == "__main__":
    test_memory_vs_baseline()
    save_memory_conversations() 
    test_multimodel_comparison()