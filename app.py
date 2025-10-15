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
# CONFIG - OPTIMIZED
# -------------------------
BASE_DIR = os.path.dirname(__file__)
FAQ_PATH = os.path.join(BASE_DIR, 'faq.json')
EMB_PATH = os.path.join(BASE_DIR, 'faq_embeddings.npy')
EMB_META = os.path.join(BASE_DIR, 'faq_emb_meta.json')

BOT_NAME = "ME Bot"
SEM_THRESHOLD = 0.55
FUZZY_THRESHOLD = 70
PHONETIC_THRESHOLD = 2
MAX_CONTEXT = 6
SAFE_FALLBACK_MAX_TOKENS = 80

# Enhanced domain keywords
DOMAIN_KEYWORDS = [
    'pcod', 'pcos', 'pregnancy', 'postpartum', 'ovary', 'ovaries', 
    'hormone', 'menstrual', 'period', 'fertility', 'symptom', 'treatment',
    'dizziness', 'hunger', 'mood', 'weight', 'insulin', 'cycle', 'pain',
    'bleeding', 'headache', 'fatigue', 'energy', 'diet', 'exercise'
]

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
# 1. FAQ TAGGING SYSTEM
# -------------------------
class FAQTaggingSystem:
    def __init__(self, faq_path):
        self.faq_path = faq_path
        self.faq_data = self.load_faq()
        self.topic_hierarchy = self.build_topic_hierarchy()
        
    def load_faq(self):
        """Load and validate FAQ data with tags"""
        with open(self.faq_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate structure
        for item in data:
            if 'tags' not in item:
                item['tags'] = self.auto_generate_tags(item['question'], item.get('topic', 'general'))
            if 'topic' not in item:
                item['topic'] = self.detect_topic(item['question'])
            if 'confidence_boosters' not in item:
                item['confidence_boosters'] = self.extract_boosters(item['question'])
                
        return data
    
    def auto_generate_tags(self, question, topic):
        """Automatically generate tags for questions"""
        question_lower = question.lower()
        tags = [topic.lower()]
        
        # Add symptom-related tags
        symptom_keywords = ['symptom', 'sign', 'experience', 'feel', 'notice', 'indication']
        if any(kw in question_lower for kw in symptom_keywords):
            tags.extend(['symptoms', 'signs'])
            
        # Add cause-related tags
        cause_keywords = ['cause', 'reason', 'why', 'trigger', 'lead to']
        if any(kw in question_lower for kw in cause_keywords):
            tags.extend(['causes', 'reasons'])
            
        # Add treatment-related tags
        treatment_keywords = ['treatment', 'cure', 'medicine', 'medication', 'therapy', 'manage']
        if any(kw in question_lower for kw in treatment_keywords):
            tags.extend(['treatment', 'management'])
            
        return list(set(tags))
    
    def detect_topic(self, question):
        """Detect main topic from question"""
        question_lower = question.lower()
        
        topic_mapping = {
            'pcod': ['pcod', 'polycystic ovarian disease'],
            'pcos': ['pcos', 'polycystic ovary syndrome'],
            'pregnancy': ['pregnancy', 'pregnant', 'gestation'],
            'postpartum': ['postpartum', 'after delivery', 'after birth'],
            'menstrual': ['period', 'menstrual', 'menstruation', 'cycle'],
            'hormonal': ['hormone', 'hormonal', 'estrogen', 'progesterone'],
            'diet': ['diet', 'food', 'nutrition', 'eat', 'meal'],
            'exercise': ['exercise', 'workout', 'yoga', 'physical activity']
        }
        
        for topic, keywords in topic_mapping.items():
            if any(kw in question_lower for kw in keywords):
                return topic
                
        return 'general'
    
    def extract_boosters(self, question):
        """Extract confidence booster words from question"""
        boosters = []
        words = word_tokenize(question.lower())
        
        booster_keywords = {
            'symptom': ['symptom', 'sign', 'experience', 'feel'],
            'cause': ['cause', 'reason', 'why', 'trigger'],
            'treatment': ['treatment', 'cure', 'medicine', 'manage'],
            'prevention': ['prevent', 'avoid', 'reduce risk'],
            'diagnosis': ['diagnose', 'test', 'detect', 'identify']
        }
        
        for category, keywords in booster_keywords.items():
            if any(kw in words for kw in keywords):
                boosters.extend(keywords)
                
        return list(set(boosters))
    
    def build_topic_hierarchy(self):
        """Build topic hierarchy for better organization"""
        hierarchy = {}
        for item in self.faq_data:
            topic = item['topic']
            if topic not in hierarchy:
                hierarchy[topic] = []
            hierarchy[topic].append(item)
        return hierarchy
    
    def get_relevant_faqs(self, user_query):
        """Get FAQs relevant to the user's query topics"""
        detected_topics = self.detect_query_topics(user_query)
        relevant_faqs = []
        
        for faq in self.faq_data:
            # Check topic match
            topic_match = faq['topic'] in detected_topics
            
            # Check tag match
            tag_match = any(tag in detected_topics for tag in faq['tags'])
            
            # Check booster match
            booster_match = any(booster in user_query.lower() for booster in faq.get('confidence_boosters', []))
            
            if topic_match or tag_match or booster_match:
                relevant_faqs.append(faq)
                
        return relevant_faqs if relevant_faqs else self.faq_data
    
    def detect_query_topics(self, user_query):
        """Detect topics from user query"""
        query_lower = user_query.lower()
        detected_topics = []
        
        topic_keywords = {
            'pcod': ['pcod', 'polycystic ovarian disease'],
            'pcos': ['pcos', 'polycystic ovary syndrome'],
            'pregnancy': ['pregnancy', 'pregnant', 'baby', 'gestation'],
            'postpartum': ['postpartum', 'after birth', 'after delivery'],
            'dizziness': ['dizzy', 'dizziness', 'lightheaded', 'vertigo'],
            'hunger': ['hungry', 'hunger', 'appetite', 'craving'],
            'mood': ['mood', 'emotional', 'feel', 'depressed', 'anxious'],
            'weight': ['weight', 'obese', 'overweight', 'bmi'],
            'diet': ['diet', 'food', 'nutrition', 'eat', 'meal'],
            'exercise': ['exercise', 'workout', 'yoga', 'walk', 'run']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(kw in query_lower for kw in keywords):
                detected_topics.append(topic)
                
        return list(set(detected_topics))

# -------------------------
# 2. CONTEXT MANAGER FIX
# -------------------------
class ConversationContextManager:
    def __init__(self):
        self.current_topic = None
        self.last_entities = []
        self.conversation_flow = []
        self.follow_up_intents = ['explain', 'clarify', 'simplify', 'more', 'detail', 'again']
        
    def extract_medical_entities(self, text):
        """Extract medical entities from text"""
        text_lower = text.lower()
        entities = []
        
        # Medical conditions
        conditions = ['pcod', 'pcos', 'pregnancy', 'postpartum', 'diabetes', 'hypertension']
        entities.extend([cond for cond in conditions if cond in text_lower])
        
        # Symptoms
        symptoms = ['dizziness', 'pain', 'bleeding', 'headache', 'fatigue', 'hunger', 'mood', 'weight']
        entities.extend([symptom for symptom in symptoms if symptom in text_lower])
        
        # Treatments
        treatments = ['diet', 'exercise', 'medication', 'yoga', 'therapy']
        entities.extend([treatment for treatment in treatments if treatment in text_lower])
        
        return list(set(entities))
    
    def is_follow_up_query(self, user_input):
        """Check if this is a follow-up query"""
        input_lower = user_input.lower()
        
        # Check for follow-up indicators
        has_follow_up_word = any(word in input_lower for word in self.follow_up_intents)
        
        # Check if it's very short (likely follow-up)
        is_short = len(word_tokenize(input_lower)) <= 4
        
        return has_follow_up_word or is_short
    
    def update_context(self, user_input, bot_response):
        """Update conversation context"""
        current_entities = self.extract_medical_entities(user_input + " " + bot_response)
        
        if current_entities:
            self.last_entities = current_entities
            self.current_topic = current_entities[0] if current_entities else None
        
        # Add to conversation flow
        self.conversation_flow.append({
            'user': user_input,
            'bot': bot_response,
            'entities': current_entities,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only recent context
        if len(self.conversation_flow) > MAX_CONTEXT:
            self.conversation_flow.pop(0)
    
    def get_context_for_follow_up(self, user_input):
        """Get relevant context for follow-up queries"""
        if not self.conversation_flow:
            return None
            
        # For follow-up queries, return the last substantial exchange
        if self.is_follow_up_query(user_input):
            for i in range(len(self.conversation_flow)-1, -1, -1):
                exchange = self.conversation_flow[i]
                if len(exchange['user'].split()) > 3:  # Substantial query
                    return exchange
                    
        return self.conversation_flow[-1] if self.conversation_flow else None

# -------------------------
# 3. SESSION MEMORY
# -------------------------
class SessionMemory:
    def __init__(self, session_id):
        self.session_id = session_id
        self.created_at = datetime.now()
        self.daily_routines = []
        self.user_experiences = []
        self.personal_preferences = {}
        self.conversation_history = []
        
    def add_conversation(self, user_input, bot_response):
        """Add conversation to history"""
        self.conversation_history.append({
            'user': user_input,
            'bot': bot_response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only recent history
        if len(self.conversation_history) > 10:
            self.conversation_history.pop(0)
    
    def add_personal_share(self, message, category):
        """Store user's personal experiences and routines"""
        share_entry = {
            'content': message,
            'category': category,
            'timestamp': datetime.now().isoformat(),
            'mood': self.extract_mood(message),
            'topics': self.extract_topics(message)
        }
        
        if category == 'daily_routine':
            self.daily_routines.append(share_entry)
        elif category == 'health_experience':
            self.user_experiences.append(share_entry)
    
    def extract_mood(self, text):
        """Extract mood from text"""
        text_lower = text.lower()
        
        mood_keywords = {
            'happy': ['happy', 'good', 'great', 'excited', 'joy', 'pleased'],
            'neutral': ['okay', 'fine', 'normal', 'alright', 'usual'],
            'stressed': ['stressed', 'anxious', 'worried', 'nervous', 'overwhelmed'],
            'tired': ['tired', 'exhausted', 'fatigued', 'sleepy', 'drained'],
            'sad': ['sad', 'depressed', 'down', 'unhappy', 'miserable']
        }
        
        for mood, keywords in mood_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return mood
                
        return 'neutral'
    
    def extract_topics(self, text):
        """Extract topics from personal shares"""
        text_lower = text.lower()
        topics = []
        
        topic_keywords = {
            'work': ['work', 'job', 'office', 'career'],
            'exercise': ['exercise', 'workout', 'gym', 'yoga', 'walk'],
            'diet': ['food', 'eat', 'meal', 'diet', 'nutrition'],
            'sleep': ['sleep', 'rest', 'tired', 'bed'],
            'social': ['friend', 'family', 'social', 'party', 'gathering'],
            'health': ['pain', 'symptom', 'feel', 'body', 'health']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(kw in text_lower for kw in keywords):
                topics.append(topic)
                
        return list(set(topics))
    
    def get_personalized_context(self):
        """Get personalized context for response generation"""
        context = {
            'recent_mood': self.daily_routines[-1]['mood'] if self.daily_routines else 'neutral',
            'frequent_topics': self.get_frequent_topics(),
            'recent_experiences': self.user_experiences[-2:] if self.user_experiences else [],
            'conversation_style': self.detect_conversation_style()
        }
        return context
    
    def get_frequent_topics(self):
        """Get frequently discussed topics"""
        all_topics = []
        for routine in self.daily_routines:
            all_topics.extend(routine['topics'])
        for experience in self.user_experiences:
            all_topics.extend(experience['topics'])
            
        from collections import Counter
        return [topic for topic, count in Counter(all_topics).most_common(3)]
    
    def detect_conversation_style(self):
        """Detect user's preferred conversation style"""
        if not self.conversation_history:
            return 'professional'
            
        # Analyze last few exchanges
        recent_exchanges = self.conversation_history[-3:]
        casual_indicators = 0
        
        for exchange in recent_exchanges:
            user_msg = exchange['user'].lower()
            if any(word in user_msg for word in ['lol', 'haha', '😊', '😂', '!']):
                casual_indicators += 1
            if len(user_msg.split()) < 4:
                casual_indicators += 1
                
        return 'casual' if casual_indicators >= 2 else 'professional'

# -------------------------
# 4. TEXT GENERATION MODULE
# -------------------------
class IntelligentResponseGenerator:
    def __init__(self, faq_system, context_manager):
        self.faq_system = faq_system
        self.context_manager = context_manager
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize GPT-2 for fallback generation
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
        self.gpt_model = GPT2LMHeadModel.from_pretrained('distilgpt2').to(self.device)
        self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token
    
    def generate_response(self, user_query, session_memory, faq_match=None):
        """Intelligently generate response using appropriate method"""
        
        # If we have a high-confidence FAQ match, use it
        if faq_match and faq_match.get('confidence', 0) > 0.7:
            return self.enhance_faq_response(faq_match, user_query, session_memory)
        
        # For follow-up queries with context
        follow_up_context = self.context_manager.get_context_for_follow_up(user_query)
        if follow_up_context and self.context_manager.is_follow_up_query(user_query):
            return self.handle_follow_up(user_query, follow_up_context, session_memory)
        
        # For personal shares
        if self.is_personal_share(user_query):
            return self.handle_personal_share(user_query, session_memory)
        
        # Generate with GPT-2 using FAQ knowledge
        return self.safe_gpt_generation(user_query, session_memory)
    
    def enhance_faq_response(self, faq_match, user_query, session_memory):
        """Enhance FAQ response with personalization and context"""
        base_answer = faq_match['answer']
        
        # Personalize based on session memory
        personal_context = session_memory.get_personalized_context()
        personalized_answer = self.add_personal_touch(base_answer, personal_context)
        
        # Add context awareness
        contextual_answer = self.add_context_awareness(personalized_answer, user_query)
        
        return contextual_answer
    
    def add_personal_touch(self, answer, personal_context):
        """Add personalization to the answer"""
        if personal_context['recent_mood'] == 'tired' and 'exercise' in answer.lower():
            return answer + " Remember to start slowly if you're feeling tired."
        elif personal_context['recent_mood'] == 'stressed' and 'diet' in answer.lower():
            return answer + " Managing stress through diet can be really helpful when you're feeling overwhelmed."
        
        return answer
    
    def add_context_awareness(self, answer, user_query):
        """Make the answer more context-aware"""
        query_lower = user_query.lower()
        
        if 'simple' in query_lower or 'explain' in query_lower:
            return self.simplify_medical_text(answer)
        elif 'detail' in query_lower or 'more about' in query_lower:
            return answer + " Would you like me to elaborate on any specific aspect?"
        
        return answer
    
    def handle_follow_up(self, user_query, context, session_memory):
        """Handle follow-up queries intelligently"""
        previous_answer = context['bot']
        
        prompt = f"""
        Previous conversation:
        User: {context['user']}
        Assistant: {previous_answer}
        
        Current follow-up question: {user_query}
        
        Provide a helpful follow-up response that builds on the previous answer:
        """
        
        return self.safe_gpt_generation_with_prompt(prompt, session_memory)
    
    def is_personal_share(self, user_query):
        """Check if this is a personal experience share"""
        personal_indicators = [
            'today i', 'i felt', 'i experienced', 'my day', 'i had',
            'i went', 'i am feeling', 'i feel', 'my routine'
        ]
        
        query_lower = user_query.lower()
        return any(indicator in query_lower for indicator in personal_indicators)
    
    def handle_personal_share(self, user_query, session_memory):
        """Handle personal experience sharing"""
        # Categorize the personal share
        if any(word in user_query.lower() for word in ['routine', 'my day', 'today i']):
            category = 'daily_routine'
        else:
            category = 'health_experience'
        
        # Store in session memory
        session_memory.add_personal_share(user_query, category)
        
        # Generate empathetic response
        prompt = f"""
        User is sharing a personal experience: "{user_query}"
        
        This appears to be about their {category}. 
        Provide an empathetic, supportive response that:
        1. Acknowledges their share
        2. Shows understanding
        3. Offers relevant support or information
        4. Maintains a caring tone
        
        Response:
        """
        
        return self.safe_gpt_generation_with_prompt(prompt, session_memory)
    
    def safe_gpt_generation(self, user_query, session_memory):
        """Safe GPT generation with healthcare constraints"""
        # Get relevant FAQ knowledge
        relevant_faqs = self.faq_system.get_relevant_faqs(user_query)
        faq_knowledge = "\n".join([f"Q: {faq['question']}\nA: {faq['answer']}" 
                                 for faq in relevant_faqs[:3]])
        
        # Get conversation history
        conv_history = session_memory.conversation_history[-3:] if session_memory.conversation_history else []
        history_text = "\n".join([f"User: {h['user']}\nAssistant: {h['bot']}" 
                                for h in conv_history])
        
        prompt = f"""
        You are ME Bot, a women's health assistant. Provide helpful, accurate information.
        
        Available knowledge:
        {faq_knowledge}
        
        Conversation history:
        {history_text}
        
        Current query: {user_query}
        
        Guidelines:
        - Be empathetic and professional
        - Stay within women's health domain
        - If unsure, suggest consulting healthcare professional
        - Use simple, clear language
        
        Response:
        """
        
        return self.generate_with_gpt(prompt)
    
    def safe_gpt_generation_with_prompt(self, prompt, session_memory):
        """Generate with custom prompt"""
        # Add safety guidelines
        safe_prompt = prompt + """
        
        Important: 
        - I am not a medical doctor
        - For serious symptoms, consult healthcare professional
        - Provide general information only
        """
        
        return self.generate_with_gpt(safe_prompt)
    
    def generate_with_gpt(self, prompt):
        """Generate text using GPT-2 with safety constraints"""
        try:
            inputs = self.gpt_tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(self.device)
            
            with torch.no_grad():
                outputs = self.gpt_model.generate(
                    **inputs,
                    max_length=min(inputs['input_ids'].shape[1] + SAFE_FALLBACK_MAX_TOKENS, 250),
                    do_sample=True,
                    top_p=0.85,
                    temperature=0.7,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.3,
                    pad_token_id=self.gpt_tokenizer.pad_token_id,
                    early_stopping=True
                )
            
            generated_text = self.gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new response
            response = generated_text[len(prompt):].strip()
            
            # Safety filtering
            if self.is_unsafe_response(response):
                return "I'm not equipped to answer that question. I specialize in women's health topics like PCOD, PCOS, pregnancy, and related health information."
            
            return response
            
        except Exception as e:
            return "I apologize, but I'm having trouble generating a response right now. Please try rephrasing your question."
    
    def is_unsafe_response(self, response):
        """Check if response contains unsafe content"""
        unsafe_indicators = [
            'prescribe', 'diagnose', 'medical advice', 'you should take',
            'definitely', 'certainly', 'http://', 'www.', '.com'
        ]
        
        response_lower = response.lower()
        return any(unsafe in response_lower for unsafe in unsafe_indicators)
    
    def simplify_medical_text(self, text):
        """Simplify medical jargon"""
        simplifications = {
            'hormonal imbalance': 'when your hormone levels are not balanced properly',
            'ovulatory dysfunction': 'when your ovaries have trouble releasing eggs',
            'metabolic abnormalities': 'when your body has trouble processing energy',
            'insulin resistance': 'when your body stops responding well to insulin',
            'polycystic ovaries': 'ovaries with many small fluid-filled sacs'
        }
        
        simple_text = text
        for complex_term, simple_explanation in simplifications.items():
            if complex_term.lower() in simple_text.lower():
                simple_text = simple_text.replace(complex_term, simple_explanation)
                
        return simple_text

# -------------------------
# MAIN CHATBOT SYSTEM
# -------------------------
class MEBot:
    def __init__(self):
        # Initialize core systems
        self.faq_system = FAQTaggingSystem(FAQ_PATH)
        self.context_manager = ConversationContextManager()
        self.response_generator = IntelligentResponseGenerator(self.faq_system, self.context_manager)
        
        # Session management
        self.active_sessions = {}
        
        # Initialize embeddings
        self.faq_embeddings = None
        self.faq_questions = [faq['question'] for faq in self.faq_system.faq_data]
        self.faq_answers = [faq['answer'] for faq in self.faq_system.faq_data]
        self.setup_embeddings()
    
    def setup_embeddings(self):
        """Setup FAQ embeddings for semantic search"""
        if os.path.exists(EMB_PATH) and os.path.exists(EMB_META):
            try:
                self.faq_embeddings = np.load(EMB_PATH)
                with open(EMB_META, 'r', encoding='utf-8') as m:
                    meta = json.load(m)
                if len(meta.get('questions', [])) != len(self.faq_questions):
                    raise Exception("FAQ changed, recomputing embeddings.")
            except Exception:
                self.faq_embeddings = None
        
        if self.faq_embeddings is None:
            print("Computing FAQ embeddings...")
            self.faq_embeddings = embed_model.encode(self.faq_questions, convert_to_numpy=True, show_progress_bar=True)
            np.save(EMB_PATH, self.faq_embeddings)
            with open(EMB_META, 'w', encoding='utf-8') as m:
                json.dump({'questions': self.faq_questions}, m)
        
        # Setup FAISS index
        if USE_FAISS:
            d = self.faq_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(d)
            norms = np.linalg.norm(self.faq_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            faq_emb_norm = self.faq_embeddings / norms
            self.index.add(faq_emb_norm.astype('float32'))
    
    def get_session(self, session_id):
        """Get or create session"""
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = SessionMemory(session_id)
        return self.active_sessions[session_id]
    
    def semantic_search(self, query, top_k=3):
        """Enhanced semantic search with topic awareness"""
        # Get relevant FAQs based on topics
        relevant_faqs = self.faq_system.get_relevant_faqs(query)
        
        if not relevant_faqs:
            relevant_faqs = self.faq_system.faq_data
        
        # Search in relevant FAQs
        relevant_questions = [faq['question'] for faq in relevant_faqs]
        relevant_indices = [i for i, faq in enumerate(self.faq_system.faq_data) if faq in relevant_faqs]
        
        if USE_FAISS:
            q_emb = embed_model.encode([query], convert_to_numpy=True)
            q_norm = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
            
            # Search in all FAQs but boost relevant ones
            D, I = self.index.search(q_norm.astype('float32'), min(top_k * 2, len(self.faq_questions)))
            
            # Boost scores for relevant FAQs
            boosted_results = []
            for i, (idx, score) in enumerate(zip(I[0], D[0])):
                if idx in relevant_indices:
                    boosted_score = min(score * 1.2, 1.0)  # Boost relevant matches
                else:
                    boosted_score = score
                
                boosted_results.append((idx, boosted_score))
            
            # Sort by boosted score and take top_k
            boosted_results.sort(key=lambda x: x[1], reverse=True)
            final_indices = [idx for idx, score in boosted_results[:top_k]]
            final_scores = [score for idx, score in boosted_results[:top_k]]
            
            return final_indices, final_scores
        else:
            # Fallback to simple search
            q_vec = embed_model.encode([query], convert_to_numpy=True)
            similarities = cosine_similarity(q_vec, self.faq_embeddings)[0]
            
            # Boost relevant FAQs
            for idx in relevant_indices:
                similarities[idx] *= 1.2
            
            top_indices = similarities.argsort()[::-1][:top_k]
            return top_indices.tolist(), similarities[top_indices].tolist()
    
    def process_query(self, user_input, session_id="default"):
        """Main method to process user queries"""
        # Get session
        session = self.get_session(session_id)
        
        # Emergency detection
        if self.detect_emergency(user_input):
            return f"🚨 EMERGENCY: {MEDICAL_DISCLAIMER} Please contact emergency services immediately."
        
        # Handle greetings and casual conversation
        casual_response = self.handle_casual_conversation(user_input)
        if casual_response:
            session.add_conversation(user_input, casual_response)
            self.context_manager.update_context(user_input, casual_response)
            return casual_response
        
        # Perform semantic search
        indices, scores = self.semantic_search(user_input)
        best_match = None
        
        if indices and scores[0] > SEM_THRESHOLD:
            best_match = {
                'answer': self.faq_answers[indices[0]],
                'confidence': scores[0],
                'question': self.faq_questions[indices[0]]
            }
        
        # Generate intelligent response
        response = self.response_generator.generate_response(user_input, session, best_match)
        
        # Update context and session
        session.add_conversation(user_input, response)
        self.context_manager.update_context(user_input, response)
        
        return response
    
    def detect_emergency(self, user_input):
        """Check for emergency keywords"""
        user_input_lower = user_input.lower()
        return any(emergency in user_input_lower for emergency in EMERGENCY_KEYWORDS)
    
    def handle_casual_conversation(self, user_input):
        """Handle casual conversation"""
        user_input_lower = user_input.strip().lower()
        
        greetings = ['hello', 'hi', 'hey', 'namaste', 'vanakam']
        if any(user_input_lower == greeting for greeting in greetings):
            return f"{BOT_NAME}: Hello! I'm ME Bot, your health assistant. How can I help you today? 😊"
        
        if user_input_lower in ['bye', 'goodbye', 'exit']:
            return f"{BOT_NAME}: Goodbye! Take care of yourself! 💖"
        
        if 'how are you' in user_input_lower:
            return f"{BOT_NAME}: I'm here and ready to help you with your health questions! How are you feeling today? 😊"
        
        return None

# -------------------------
# FLASK APP & CLI
# -------------------------
app = Flask(__name__)
CORS(app)
me_bot = MEBot()

@app.route('/api/chat', methods=['POST'])
def api_chat():
    body = request.json or {}
    user_input = body.get('message', '').strip()
    session_id = body.get('session_id', 'default')
    
    if not user_input:
        return jsonify({'response': f"{BOT_NAME}: Please send a non-empty message."})
    
    resp = me_bot.process_query(user_input, session_id)
    return jsonify({'response': resp})

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'bot_name': BOT_NAME,
        'faq_count': len(me_bot.faq_questions),
        'active_sessions': len(me_bot.active_sessions)
    })

def run_cli():
    print(f"{BOT_NAME} Advanced Version")
    print("=" * 50)
    print("Features: FAQ Tagging | Context Management | Session Memory | Intelligent Generation")
    print("=" * 50)
    
    session_id = str(uuid.uuid4())[:8]
    print(f"Session ID: {session_id}")
    print("\nYou can:")
    print("- Ask health questions")
    print("- Share daily experiences ('Today I felt...')") 
    print("- Ask for explanations ('explain simply')")
    print("- Type 'bye' to exit")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print(f"{BOT_NAME}: Goodbye! Take care! 💖")
                break
                
            start_time = time.time()
            response = me_bot.process_query(user_input, session_id)
            end_time = time.time()
            
            print(response)
            print(f"[Processed in {end_time-start_time:.2f}s]")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print(f"\n{BOT_NAME}: Session ended. Stay healthy! 💪")
            break
        except Exception as e:
            print(f"{BOT_NAME}: Sorry, I encountered an error: {e}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--cli':
        run_cli()
    else:
        print(f"{BOT_NAME} Advanced API starting on port 5000")
        app.run(port=5000, debug=False)
