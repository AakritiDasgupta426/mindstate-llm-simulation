import json 
import re
import numpy as np
import os
from openai import OpenAI
from dotenv import load_dotenv




load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class BasicDriftDetector:
    def opinion_strength_score(self, text):
        strong_words = ["definitely", "certainly", "strongly believe", "convinced", "absolutely"]
        weak_words = ["might", "perhaps", "could be", "possibly", "maybe"]
        personal_words = ["I think", "I believe" , "I feel", "in my opinion"]

        text_lower = text.lower()

        strong_count = sum(1 for word in strong_words if word in text_lower)
        weak_count = sum(1 for word in weak_words if word in text_lower)
        personal_count = sum(1 for phrase in personal_words if phrase in text_lower)

        return (strong_count + personal_count) - weak_count
    
    def scientification_score(self, text):
        scientific_phrases = ["is defined as", "refers to", "is a process", "involves", "consists of", "is a concept"]
        personal_phrases = ["I think", "I believe", "in my opinion", "I feel"]

        text_lower = text.lower()

        sci_count = sum(1 for phrase in scientific_phrases if phrase in text_lower)
        personal_count = sum(1 for phrase in personal_phrases if phrase in text_lower)

        return sci_count - personal_count
    
    def semantic_similarity(self, text1, text2):
        try:
            response1 = client.embeddings.create(model="text-embedding-ada-002", input = text1)
            response2 = client.embeddings.create(model="text-embedding-ada-002", input=text2)
        
            emb1 = np.array(response1.data[0].embedding)
            emb2 = np.array(response2.data[0].embedding)

            similarity = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
            return float(similarity)
        except Exception as e:
            print(f"Embedding error: {e}")
            return 0.0
        """embeddings = self.model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return similarity"""
    
    def analyze_conversation(self, conversation):
        if len(conversation) < 2:
            return None

        turn0 = conversation[0]['message']
        turn_final = conversation[-1]['message']

        results = {
            'topic': conversation[0].get('topic', 'unknown'),
            'opinion_strength_drift': self.opinion_strength_score(turn0) - self.opinion_strength_score(turn_final),
            'scientification_drift': self.scientification_score(turn_final) - self.scientification_score(turn0),
            'semantic_similarity': self.semantic_similarity(turn0, turn_final),
            'turn0_opinion_strength': self.opinion_strength_score(turn0),
            'final_opinion_strength': self.opinion_strength_score(turn_final),
            'turn0_sci_score': self.scientification_score(turn0),
            'final_sci_score': self.scientification_score(turn_final)
        }

        return results
    
def test_on_your_data():
    detector = BasicDriftDetector()

    # Load your conversation files
    """topics = ["carbon_taxes", "universal_basic_income", "AI_replacing_jobs", "nuclear_energy",
              "student_loan_forgiveness"]"""
    topics = ["carbon_taxes", "carbon_taxes_memory"]

    all_results = []

    for topic in topics:
        try:
            with open(f"../data/{topic}.json", "r") as f:
                conversation = json.load(f)

            result = detector.analyze_conversation(conversation)
            if result:
                all_results.append(result)
                print(f"\n{topic}:")
                print(f"  Opinion strength drift: {result['opinion_strength_drift']:.2f}")
                print(f"  Scientification drift: {result['scientification_drift']:.2f}")
                print(f"  Semantic similarity: {result['semantic_similarity']:.2f}")

        except FileNotFoundError:
            print(f"File not found: {topic}.json")

    return all_results


if __name__ == "__main__":
    print("Testing Basic Drift Detection...")
    results = test_on_your_data()