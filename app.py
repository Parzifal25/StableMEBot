import os
import json
import time
import numpy as np
from datetime import datetime
import gc
from functools import lru_cache
import re
import uuid

# models/nlp
from sentence_transformers import SentenceTransformer
from spellchecker import SpellChecker
import phonetics
from fuzzywuzzy import fuzz

# vector search fallback
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# transformers for safe fallback
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# web and NLP utils
from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------------
# CONFIG
# -------------------------
BASE_DIR = os.path.dirname(__file__)
TRANSFORMED_FAQ_PATH = os.path.join(BASE_DIR, 'data', 'transformed_faq.json')
ORIGINAL_FAQ_PATH = os.path.join(BASE_DIR, 'faq.json')
EMB_PATH = os.path.join(BASE_DIR, 'faq_embeddings.npy')
EMB_META = os.path.join(BASE_DIR, 'faq_emb_meta.json')

BOT_NAME = "ME Bot"
SEM_THRESHOLD = 0.55
FUZZY_THRESHOLD = 70
PHONETIC_THRESHOLD = 2
MAX_CONTEXT = 6
SAFE_FALLBACK_MAX_TOKENS = 80

MEDICAL_DISCLAIMER = "⚠️ I'm an AI assistant, not a medical professional. For serious symptoms, please consult a doctor immediately."
EMERGENCY_KEYWORDS = ['chest pain', 'difficulty breathing', 'heavy bleeding', 'suicide', 'emergency', 'urgent help']

# -------------------------
# INIT: NLP tools
# -------------------------
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

STOPWORDS = set(stopwords.words('english'))
LEMMA = WordNetLemmatizer()

SPELL = SpellChecker()
EMBED_MODEL_NAME = 'all-MiniLM-L6-v2'
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

USE_FAISS = FAISS_AVAILABLE

# -------------------------
# FAQ TAGGING SYSTEM
# -------------------------
class FAQTaggingSystem:
    def __init__(self):
        self.faq_data = self.load_faq_with_fallback()
        
    def load_faq_with_fallback(self):
        """Load transformed FAQ or fallback to original"""
        if os.path.exists(TRANSFORMED_FAQ_PATH):
            try:
                with open(TRANSFORMED_FAQ_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"✅ Loaded transformed FAQ with {len(data)} entries")
                return data
            except Exception as e:
                print(f"❌ Error loading transformed FAQ: {e}")
        
        if os.path.exists(ORIGINAL_FAQ_PATH):
            try:
                with open(ORIGINAL_FAQ_PATH, 'r', encoding='utf-8') as f:
                    original_data = json.load(f)
                print(f"🔄 Using original FAQ with {len(original_data)} entries")
                return self.convert_original_to_transformed(original_data)
            except Exception as e:
                print(f"❌ Error loading original FAQ: {e}")
        
        print("❌ No FAQ data found! Using empty dataset.")
        return []
    
    def convert_original_to_transformed(self, original_data):
        """Convert original FAQ format to transformed format"""
        transformed = []
        for item in original_data:
            question = item.get('question', '')
            answer = item.get('answer', '')
            
            topic, subtopic, tags, boosters = self.detect_topic_fallback(question)
            
            transformed.append({
                "topic": topic,
                "subtopic": subtopic,
                "tags": tags,
                "question": question,
                "answer": answer,
                "confidence_boosters": boosters
            })
        return transformed
    
    def detect_topic_fallback(self, question):
        """Basic topic detection for fallback"""
        q_lower = question.lower()
        
        if any(k in q_lower for k in ['pcod', 'pcos', 'polycystic']):
            return "PCOD / PCOS Basics", "general", ["pcod", "pcos"], ["pcod", "pcos"]
        elif any(k in q_lower for k in ['period', 'menstrual', 'cycle']):
            return "Menstrual Health", "cycle", ["period", "menstrual"], ["period", "cycle"]
        elif any(k in q_lower for k in ['pregnancy', 'fertility']):
            return "Fertility & Pre-pregnancy", "conception", ["pregnancy", "fertility"], ["pregnant", "fertile"]
        else:
            return "General FAQs", "general", ["faq"], ["question"]
    
    def get_relevant_faqs(self, user_query):
        """Get FAQs relevant to the user's query topics"""
        if not self.faq_data:
            return []
            
        detected_topics = self.detect_query_topics(user_query)
        relevant_faqs = []
        
        for faq in self.faq_data:
            topic_match = faq.get('topic', '').lower() in [t.lower() for t in detected_topics]
            tags = faq.get('tags', [])
            tag_match = any(tag.lower() in user_query.lower() for tag in tags)
            boosters = faq.get('confidence_boosters', [])
            booster_match = any(booster in user_query.lower() for booster in boosters)
            
            if topic_match or tag_match or booster_match:
                relevant_faqs.append(faq)
                
        return relevant_faqs if relevant_faqs else self.faq_data
    
    def detect_query_topics(self, user_query):
        """Detect topics from user query"""
        query_lower = user_query.lower()
        detected_topics = []
        
        topic_mapping = {
            'PCOD / PCOS Basics': ['pcod', 'pcos', 'polycystic', 'ovarian'],
            'Menstrual Health': ['period', 'menstrual', 'cycle', 'bleeding', 'pms', 'cramps'],
            'Fertility & Pre-pregnancy': ['fertility', 'pregnant', 'pregnancy', 'ovulate', 'conceive'],
            'Hormonal & Emotional Wellness': ['hormone', 'stress', 'anxiety', 'mood', 'emotional', 'mental'],
            'Women\'s General Health': ['thyroid', 'metabolism', 'energy', 'general health', 'wellness'],
            'Diet & Nutrition': ['food', 'diet', 'nutrition', 'eat', 'meal', 'weight']
        }
        
        for topic, keywords in topic_mapping.items():
            if any(kw in query_lower for kw in keywords):
                detected_topics.append(topic)
                
        return list(set(detected_topics))

# -------------------------
# SIMPLIFIED RESPONSE GENERATOR
# -------------------------
class SimpleResponseGenerator:
    def __init__(self, faq_system):
        self.faq_system = faq_system
        
    def generate_safe_response(self, user_query, faq_match=None):
        """Generate safe, relevant responses"""
        
        # If we have a good FAQ match, use it
        if faq_match and faq_match.get('confidence', 0) > 0.6:
            return self.enhance_faq_response(faq_match, user_query)
        
        # For simple greetings and casual talk
        casual_response = self.handle_casual_query(user_query)
        if casual_response:
            return casual_response
        
        # Try to find relevant FAQ manually
        manual_match = self.find_manual_match(user_query)
        if manual_match:
            return manual_match
        
        # Default response
        return "I specialize in women's health topics like PCOD, PCOS, menstrual health, pregnancy, and nutrition. Could you ask me something specific in these areas?"
    
    def enhance_faq_response(self, faq_match, user_query):
        """Enhance FAQ response naturally"""
        base_answer = faq_match['answer']
        
        if any(word in user_query.lower() for word in ['worried', 'concerned', 'scared']):
            return f"I understand this might be concerning. {base_answer} Remember, many women experience this and there are ways to manage it effectively."
        elif any(word in user_query.lower() for word in ['explain', 'simple', 'easier']):
            return f"Let me explain this clearly: {base_answer}"
        else:
            return base_answer
    
    def handle_casual_query(self, user_query):
        """Handle casual conversation"""
        user_input_lower = user_query.lower().strip()
        
        greetings = ['hello', 'hi', 'hey', 'namaste', 'vanakam']
        if any(user_input_lower == greeting for greeting in greetings):
            return f"{BOT_NAME}: Hello! I'm ME Bot, your health assistant. How can I help you today? 😊"
        
        if user_input_lower in ['bye', 'goodbye', 'exit']:
            return f"{BOT_NAME}: Goodbye! Take care of yourself! 💖"
        
        if 'how are you' in user_input_lower:
            return f"{BOT_NAME}: I'm here and ready to help you with your health questions! How are you feeling today? 😊"
        
        if any(word in user_input_lower for word in ['thank', 'thanks']):
            return f"{BOT_NAME}: You're welcome! I'm glad I could help. 😊"
        
        return None
    
    def find_manual_match(self, user_query):
        """Manual keyword-based matching as fallback"""
        query_lower = user_query.lower()
        
        # Common women's health questions and answers
        manual_responses = {
            'cramps': "For menstrual cramps, try gentle exercise, heat pads, staying hydrated, and over-the-counter pain relief if appropriate. If severe, consult your doctor.",
            'pcos food': "For PCOS, focus on anti-inflammatory foods: leafy greens, fatty fish, berries, nuts. Avoid processed foods and sugars. Balanced diet helps manage symptoms.",
            'pcod diet': "With PCOD, eat high-fiber foods, lean proteins, and healthy fats. Small frequent meals help manage insulin levels. Consult a nutritionist for personalized advice.",
            'period pain': "For period pain, try heat therapy, light exercise, staying hydrated, and relaxation techniques. If pain is severe, consult your healthcare provider.",
            'irregular periods': "Irregular periods can be caused by stress, weight changes, or hormonal issues. Tracking cycles and maintaining healthy lifestyle often helps.",
            'weight gain pcos': "PCOS can make weight management challenging. Focus on balanced diet, regular exercise, and stress management. Consult healthcare provider for guidance.",
            'mood swings': "Hormonal changes can affect mood. Regular sleep, balanced diet, exercise, and stress management techniques can help. Seek support if needed.",
            'fertility': "For fertility concerns, track ovulation, maintain healthy weight, avoid smoking/alcohol, and consider consulting a fertility specialist.",
            'hormonal imbalance': "Hormonal balance can be supported through stress management, balanced nutrition, regular exercise, and adequate sleep. Consult doctor for specific concerns."
        }
        
        for keyword, response in manual_responses.items():
            if keyword in query_lower:
                return response
        
        return None

# -------------------------
# MAIN CHATBOT WITH FIXED EMBEDDINGS
# -------------------------
class SimpleMEBot:
    def __init__(self):
        print("🚀 Initializing ME Bot...")
        
        # Initialize core systems
        self.faq_system = FAQTaggingSystem()
        self.response_generator = SimpleResponseGenerator(self.faq_system)
        
        # Session tracking
        self.conversation_history = []
        
        # Initialize embeddings
        self.setup_embeddings()
        
        print("✅ ME Bot initialized successfully!")
    
    def setup_embeddings(self):
        """Setup FAQ embeddings with proper validation"""
        self.faq_questions = [faq['question'] for faq in self.faq_system.faq_data]
        self.faq_answers = [faq['answer'] for faq in self.faq_system.faq_data]
        
        if not self.faq_questions:
            print("❌ No FAQ questions found! Using basic mode without embeddings.")
            self.faq_embeddings = None
            return
        
        print(f"🔄 Setting up embeddings for {len(self.faq_questions)} questions...")
        
        try:
            # Always recompute embeddings to ensure they're valid
            print("🔄 Computing fresh embeddings...")
            self.faq_embeddings = embed_model.encode(self.faq_questions, convert_to_numpy=True, show_progress_bar=True)
            
            # Validate embeddings
            if (self.faq_embeddings is not None and 
                len(self.faq_embeddings.shape) == 2 and 
                self.faq_embeddings.shape[0] == len(self.faq_questions)):
                
                np.save(EMB_PATH, self.faq_embeddings)
                print(f"✅ Embeddings computed and saved: {self.faq_embeddings.shape}")
                
                # Setup FAISS if available
                if USE_FAISS:
                    d = self.faq_embeddings.shape[1]
                    self.index = faiss.IndexFlatIP(d)
                    norms = np.linalg.norm(self.faq_embeddings, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    faq_emb_norm = self.faq_embeddings / norms
                    self.index.add(faq_emb_norm.astype('float32'))
                    print("✅ FAISS index built")
                else:
                    self.index = None
                    print("ℹ️ FAISS not available - using basic search")
            else:
                print("❌ Embeddings computation failed")
                self.faq_embeddings = None
                self.index = None
                
        except Exception as e:
            print(f"❌ Error setting up embeddings: {e}")
            self.faq_embeddings = None
            self.index = None
    
    def semantic_search(self, query, top_k=3):
        """Robust semantic search with error handling"""
        if (self.faq_embeddings is None or 
            not hasattr(self, 'faq_questions') or 
            not self.faq_questions):
            return [], []
            
        try:
            # Encode query
            q_emb = embed_model.encode([query], convert_to_numpy=True)
            
            # Ensure query embedding is 2D
            if len(q_emb.shape) == 1:
                q_emb = q_emb.reshape(1, -1)
            
            # Normalize query
            q_norm = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
            
            if USE_FAISS and hasattr(self, 'index') and self.index is not None:
                # FAISS search
                D, I = self.index.search(q_norm.astype('float32'), top_k)
                return I[0].tolist(), D[0].tolist()
            else:
                # Basic cosine similarity
                # Ensure FAQ embeddings are 2D
                if len(self.faq_embeddings.shape) == 1:
                    faq_emb_2d = self.faq_embeddings.reshape(1, -1)
                else:
                    faq_emb_2d = self.faq_embeddings
                
                similarities = cosine_similarity(q_norm, faq_emb_2d)[0]
                top_indices = similarities.argsort()[::-1][:top_k]
                return top_indices.tolist(), similarities[top_indices].tolist()
                
        except Exception as e:
            print(f"❌ Search error: {e}")
            return [], []
    
    def process_query(self, user_input):
        """Main method to process user queries"""
        # Emergency detection
        if self.detect_emergency(user_input):
            return f"🚨 EMERGENCY: {MEDICAL_DISCLAIMER} Please contact emergency services immediately."
        
        # Handle casual conversation first
        casual_response = self.response_generator.handle_casual_query(user_input)
        if casual_response:
            self.conversation_history.append({'user': user_input, 'bot': casual_response})
            if len(self.conversation_history) > MAX_CONTEXT:
                self.conversation_history.pop(0)
            return casual_response
        
        # Try semantic search
        indices, scores = self.semantic_search(user_input)
        best_match = None
        
        if indices and scores and scores[0] > SEM_THRESHOLD:
            best_match = {
                'answer': self.faq_answers[indices[0]],
                'confidence': scores[0],
                'question': self.faq_questions[indices[0]]
            }
            print(f"🔍 Found semantic match: {best_match['confidence']:.3f}")
        
        # Generate response
        response = self.response_generator.generate_safe_response(user_input, best_match)
        
        # Update conversation history
        self.conversation_history.append({'user': user_input, 'bot': response})
        if len(self.conversation_history) > MAX_CONTEXT:
            self.conversation_history.pop(0)
        
        return response
    
    def detect_emergency(self, user_input):
        """Check for emergency keywords"""
        user_input_lower = user_input.lower()
        return any(emergency in user_input_lower for emergency in EMERGENCY_KEYWORDS)

# -------------------------
# FLASK APP & CLI
# -------------------------
app = Flask(__name__)
CORS(app)

# Initialize the bot
try:
    me_bot = SimpleMEBot()
    bot_initialized = True
except Exception as e:
    print(f"❌ Failed to initialize bot: {e}")
    bot_initialized = False
    me_bot = None

@app.route('/api/chat', methods=['POST'])
def api_chat():
    if not bot_initialized or me_bot is None:
        return jsonify({'response': f"{BOT_NAME}: Service temporarily unavailable. Please try again later."})
    
    body = request.json or {}
    user_input = body.get('message', '').strip()
    
    if not user_input:
        return jsonify({'response': f"{BOT_NAME}: Please send a non-empty message."})
    
    resp = me_bot.process_query(user_input)
    return jsonify({'response': resp})

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy' if bot_initialized else 'unhealthy',
        'bot_name': BOT_NAME,
        'faq_count': len(me_bot.faq_questions) if bot_initialized and hasattr(me_bot, 'faq_questions') else 0,
        'embeddings_loaded': me_bot.faq_embeddings is not None if bot_initialized else False
    })

def run_cli():
    if not bot_initialized:
        print("❌ Bot failed to initialize. Please check your FAQ files.")
        return
    
    print(f"{BOT_NAME} Fixed Version")
    print("=" * 50)
    print("Ask me about: PCOD, PCOS, menstrual health, pregnancy, hormones, diet")
    print("Type 'bye' to exit")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print(f"{BOT_NAME}: Goodbye! Take care! 💖")
                break
                
            start_time = time.time()
            response = me_bot.process_query(user_input)
            end_time = time.time()
            
            print(response)
            print(f"[Processed in {end_time-start_time:.2f}s]")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print(f"\n{BOT_NAME}: Session ended. Stay healthy! 💪")
            break
        except Exception as e:
            print(f"{BOT_NAME}: Sorry, I encountered an error. Please try again.")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--cli':
        run_cli()
    else:
        if bot_initialized:
            print(f"{BOT_NAME} API starting on port 5000")
            app.run(port=5000, debug=False)
        else:
            print(f"❌ {BOT_NAME} failed to start. Please check your FAQ files.")
