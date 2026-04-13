import openai
import json
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_conversations(topic, turns = 10):
    messages = [{"role": "system", "content":f"You are having a casual conversation about {topic}. Express your thoughts and beliefs clearly."}]

    conversations = []
    for i in range(turns):
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens = 150
        )

        content = response.choices[0].message.content
        turn_data = {
            "turn" : i,
            "message" : content,
            "timestamp" : datetime.now().isoformat()
        }
        conversations.append(turn_data)

        messages.append({"role":"assistant", "content": content})
        messages.append({"role":"user", "content": f"That is interesting. Tell me more about {topic}."})
        
    return conversations

def generate_belief_dataset():
    topics = [
        "carbon taxes",
        "universal basic income",
        "AI replacing jobs",
        "nuclear energy",
        "student loan forgiveness"
    ]

    all_conversations = {}

    for topic in topics:
        print(f"Generating conversation about : {topic}")
        conv = generate_conversations(topic, turns = 3)

        filename = topic.replace(' ', '_').replace(',', '')
        with open(f"../data/{filename}.json", "w") as f:
            json.dump(conv, f, indent = 2)
        
        all_conversations[topic] = conv
        print(f"Saved {filename}.json")

    with open("../data/all_belief_conversations.json", "w") as f:
        json.dump(all_conversations, f, indent=2)
    
    print(f"\n Generated {len(topics)} belief conversations!")
    return all_conversations

if __name__ == "__main__":
    conversations = generate_belief_dataset()
