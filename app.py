import os
import json
import time
import numpy as np
from datetime import datetime
import re
import uuid
import hashlib
from collections import deque
import faiss
from sentence_transformers import SentenceTransformer
import torch
from transformers import pipeline
from rapidfuzz import fuzz
import phonetics
import sys

# ================================
# OPTIMIZED CONFIGURATION
# ================================
BASE_DIR = os.path.dirname(__file__)
FAQ_PATH = os.path.join(BASE_DIR, 'faq.json')
KNOWLEDGE_GRAPH_PATH = os.path.join(BASE_DIR, 'knowledge_graph.json')
USER_PROFILES_PATH = os.path.join(BASE_DIR, 'user_profiles.json')

BOT_NAME = "ME Bot Pro"
MAX_CONTEXT = 10  # Reduced for better performance
MEDICAL_DISCLAIMER = "⚠️ I'm an AI assistant, not a medical professional. For serious symptoms, please consult a doctor immediately."

# Optimized emergency detection
CRITICAL_EMERGENCIES = {
    'chest pain', 'heart attack', 'stroke', 'cannot breathe', 
    'unconscious', 'suicidal', 'self harm', 'want to die'
}

URGENT_CONDITIONS = {
    'severe bleeding', 'heavy bleeding', 'severe pain', 'excruciating pain',
    'difficulty breathing', 'fainting', 'seizure'
}

CRISIS_KEYWORDS = {
    'suicidal', 'kill myself', 'end it all', 'want to die', 'self harm'
}

# ================================
# OPTIMIZED SMART NLP
# ================================

class SmartNLP:
    """Optimized NLP with better intent detection"""
    
    MEDICAL_SYNONYMS = {
        'pcod': {'pcos', 'polycystic', 'ovarian cyst'},
        'period': {'menstrual', 'menstruation', 'monthly', 'cycle'},
        'hormone': {'hormonal', 'estrogen', 'progesterone'},
        'stress': {'anxiety', 'tension', 'pressure'},
        'diet': {'nutrition', 'food', 'eating'},
        'exercise': {'workout', 'fitness', 'physical'}
    }
    
    INTENT_PATTERNS = {
        'definition': {'what is', 'define', 'explain', 'meaning of'},
        'symptoms': {'symptoms', 'signs', 'experience', 'feel like'},
        'treatment': {'treatment', 'cure', 'medicine', 'how to treat', 'management'},
        'causes': {'causes', 'reasons', 'why', 'what causes'},
        'prevention': {'prevent', 'avoid', 'reduce risk'}
    }
    
    @staticmethod
    def analyze_query(query):
        """Fast and accurate query analysis"""
        query_lower = query.lower().strip()
        
        # Quick intent detection
        intent = 'general'
        for intent_type, patterns in SmartNLP.INTENT_PATTERNS.items():
            if any(pattern in query_lower for pattern in patterns):
                intent = intent_type
                break
        
        # Medical context detection
        medical_context = []
        for primary, synonyms in SmartNLP.MEDICAL_SYNONYMS.items():
            if (primary in query_lower or 
                any(synonym in query_lower for synonym in synonyms)):
                medical_context.append(primary)
        
        # Tone analysis (optimized)
        tone = 'neutral'
        tone_words = {
            'anxious': {'worried', 'scared', 'anxious', 'nervous'},
            'grateful': {'thank', 'thanks', 'appreciate'},
            'confused': {'confused', 'unsure', 'don\'t understand'},
            'frustrated': {'frustrated', 'annoyed', 'not helping'}
        }
        
        for tone_type, words in tone_words.items():
            if any(word in query_lower for word in words):
                tone = tone_type
                break
        
        return {
            'intent': intent,
            'medical_context': medical_context,
            'tone': tone,
            'phonetic_variations': SmartNLP._get_phonetic_variations(query)
        }
    
    @staticmethod
    def _get_phonetic_variations(query):
        """Generate phonetic variations for typo tolerance"""
        words = query.split()
        phonetic_variants = []
        
        for word in words:
            if len(word) > 3:
                try:
                    phonetic = phonetics.metaphone(word)
                    if phonetic:
                        phonetic_variants.append(phonetic)
                except:
                    continue
        return phonetic_variants

# ================================
# OPTIMIZED MEDICAL NLP
# ================================

class AdvancedMedicalNLP:
    """Fast medical entity recognition"""
    
    # Optimized entity sets
    CONDITIONS = {'pcod', 'pcos', 'endometriosis', 'fibroids', 'thyroid', 'diabetes'}
    SYMPTOMS = {'pain', 'bleeding', 'cramps', 'headache', 'nausea', 'fatigue'}
    TREATMENTS = {'medication', 'treatment', 'therapy', 'diet', 'exercise'}
    
    @staticmethod
    def extract_medical_entities(text):
        """Fast entity extraction"""
        text_lower = text.lower()
        entities = {}
        
        # Quick entity checks
        entities['conditions'] = [cond for cond in AdvancedMedicalNLP.CONDITIONS if cond in text_lower]
        entities['symptoms'] = [symptom for symptom in AdvancedMedicalNLP.SYMPTOMS if symptom in text_lower]
        entities['treatments'] = [treatment for treatment in AdvancedMedicalNLP.TREATMENTS if treatment in text_lower]
        
        # Clean empty entities
        entities = {k: v for k, v in entities.items() if v}
        
        return entities

# ================================
# OPTIMIZED SEARCH SYSTEM
# ================================

class OptimizedSearch:
    """Fast and accurate search with multiple strategies"""
    
    def __init__(self, faq_data):
        self.faq_data = faq_data
        self.questions, self.answers = self._extract_qa_pairs(faq_data)
        print(f"✅ Prepared {len(self.questions)} Q&A pairs")
        
        # Initialize embedder
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self._setup_semantic_index()
    
    def _extract_qa_pairs(self, faq_data):
        """Efficient QA extraction"""
        questions, answers = [], []
        
        for item in faq_data:
            if isinstance(item, dict):
                if 'question' in item and 'answer' in item:
                    questions.append(item['question'])
                    answers.append(item['answer'])
                elif 'q' in item and 'a' in item:
                    questions.append(item['q'])
                    answers.append(item['a'])
                else:
                    # Fallback: use first two values
                    values = list(item.values())
                    if len(values) >= 2:
                        questions.append(str(values[0]))
                        answers.append(str(values[1]))
            else:
                questions.append(str(item))
                answers.append(str(item))
        
        return questions, answers
    
    def _setup_semantic_index(self):
        """Setup FAISS index efficiently"""
        try:
            if not self.questions:
                self.index = None
                return
                
            print("🔄 Creating semantic index...")
            embeddings = self.embedder.encode(self.questions, show_progress_bar=False)
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            
            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings_norm = embeddings / norms
            self.index.add(embeddings_norm.astype('float32'))
            
            print(f"✅ Semantic index ready with {len(self.questions)} vectors")
        except Exception as e:
            print(f"⚠️ Index setup failed: {e}")
            self.index = None
    
    def intelligent_search(self, query, nlp_analysis):
        """Fast multi-strategy search"""
        all_matches = []
        
        # 1. Semantic search (primary)
        semantic_matches = self._semantic_search(query)
        all_matches.extend(semantic_matches)
        
        # 2. Keyword search (fallback)
        if not all_matches or all_matches[0]['score'] < 0.5:
            keyword_matches = self._keyword_search(query, nlp_analysis)
            all_matches.extend(keyword_matches)
        
        return self._rank_matches(all_matches, nlp_analysis)
    
    def _semantic_search(self, query, top_k=5):
        """Fast semantic search"""
        if not self.index:
            return []
        
        try:
            query_vec = self.embedder.encode([query])
            query_norm = query_vec / np.linalg.norm(query_vec)
            
            scores, indices = self.index.search(query_norm.astype('float32'), top_k)
            
            matches = []
            for idx, score in zip(indices[0], scores[0]):
                if score > 0.3:  # Reasonable similarity threshold
                    matches.append({
                        'index': idx,
                        'score': float(score),
                        'question': self.questions[idx],
                        'answer': self.answers[idx],
                        'type': 'semantic'
                    })
            return matches
        except Exception as e:
            return []
    
    def _keyword_search(self, query, nlp_analysis):
        """Fast keyword-based search"""
        query_lower = query.lower()
        matches = []
        
        for idx, (question, answer) in enumerate(zip(self.questions, self.answers)):
            question_lower = question.lower()
            
            # Check query terms in question
            matches_count = sum(1 for word in query_lower.split() 
                              if len(word) > 3 and word in question_lower)
            
            if matches_count > 0:
                score = min(matches_count / len(query.split()) * 0.8, 0.8)
                matches.append({
                    'index': idx,
                    'score': score,
                    'question': question,
                    'answer': answer,
                    'type': 'keyword'
                })
        
        return matches
    
    def _rank_matches(self, matches, nlp_analysis):
        """Intelligent ranking"""
        if not matches:
            return []
        
        # Remove duplicates
        unique_matches = {}
        for match in matches:
            key = match['question']
            if key not in unique_matches or match['score'] > unique_matches[key]['score']:
                unique_matches[key] = match
        
        # Sort by score
        sorted_matches = sorted(unique_matches.values(), 
                              key=lambda x: x['score'], reverse=True)
        
        return sorted_matches[:3]  # Return top 3

# ================================
# OPTIMIZED RESPONSE BUILDER
# ================================

class OptimizedResponseBuilder:
    """Fast and intelligent response generation"""
    
    def __init__(self):
        self.nlp = SmartNLP()
    
    def build_response(self, query, matches, conversation_context, medical_entities):
        """Build optimal response"""
        if not matches:
            return self._intelligent_fallback(query, conversation_context)
        
        # Get best match
        best_match = matches[0]
        base_response = best_match['answer']
        
        # Ensure response quality
        if len(base_response.strip()) < 25 or base_response.strip() == query:
            if len(matches) > 1:
                base_response = matches[1]['answer']
            else:
                return self._intelligent_fallback(query, conversation_context)
        
        # Add context if available
        response = self._add_context(base_response, conversation_context)
        
        # Add tone adaptation
        response = self._adapt_tone(response, conversation_context)
        
        return response
    
    def _add_context(self, response, conversation_context):
        """Add conversation context naturally"""
        if conversation_context and conversation_context.get('topics'):
            recent_topics = conversation_context['topics'][-2:]
            if recent_topics:
                # Use user-friendly topic names
                friendly_topics = []
                for topic in recent_topics:
                    if topic == 'pcod': friendly_topics.append('PCOD')
                    elif topic == 'pcos': friendly_topics.append('PCOS')
                    elif topic == 'period': friendly_topics.append('period concerns')
                    elif topic == 'hormone': friendly_topics.append('hormonal balance')
                    elif topic == 'stress': friendly_topics.append('stress management')
                    else: friendly_topics.append(topic)
                
                topic_str = ' and '.join(friendly_topics)
                return f"Continuing our discussion about {topic_str}: {response}"
        
        return response
    
    def _adapt_tone(self, response, conversation_context):
        """Adapt to user's emotional tone"""
        if conversation_context and conversation_context.get('recent_history'):
            last_user_msg = conversation_context['recent_history'][-1].get('user', '').lower()
            
            if any(word in last_user_msg for word in ['frustrated', 'annoyed', 'not helping']):
                return f"I understand this might be frustrating. {response}"
            elif any(word in last_user_msg for word in ['confused', 'don\'t understand']):
                return f"Let me clarify this simply: {response}"
            elif any(word in last_user_msg for word in ['worried', 'scared', 'anxious']):
                return f"I understand this might be concerning. {response}"
        
        return response
    
    def _intelligent_fallback(self, query, conversation_context):
        """Smart fallback responses"""
        nlp_analysis = self.nlp.analyze_query(query)
        
        if nlp_analysis['medical_context']:
            topics = ', '.join(nlp_analysis['medical_context'])
            return f"I specialize in {topics} and related women's health topics. Could you provide more details about your specific concern?"
        
        return "I focus on women's health including PCOD, PCOS, menstrual health, and hormonal balance. What specific area would you like to know about?"

# ================================
# OPTIMIZED SESSION MANAGER
# ================================

class SessionManager:
    """Efficient session management"""
    
    def __init__(self):
        self.sessions = {}
    
    def get_session(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'history': deque(maxlen=MAX_CONTEXT),
                'topics': set(),
                'created': datetime.now(),
                'last_activity': datetime.now()
            }
        return self.sessions[session_id]
    
    def update_session(self, session_id, user_input, response, medical_context):
        session = self.get_session(session_id)
        
        # Update history
        session['history'].append({
            'user': user_input,
            'bot': response,
            'time': datetime.now()
        })
        
        # Update topics (only meaningful medical contexts)
        meaningful_topics = [topic for topic in medical_context 
                           if topic in ['pcod', 'pcos', 'period', 'hormone', 'stress', 'diet', 'exercise']]
        session['topics'].update(meaningful_topics)
        
        session['last_activity'] = datetime.now()
    
    def get_context(self, session_id):
        session = self.get_session(session_id)
        return {
            'recent_history': list(session['history'])[-3:],
            'topics': list(session['topics'])[-5:]  # Recent topics only
        }

# ================================
# OPTIMIZED MAIN BOT
# ================================

class OptimizedMEBot:
    """Fast, reliable medical AI assistant"""
    
    def __init__(self):
        print("🚀 Initializing Optimized ME Bot...")
        
        # Load data
        self.faq_data = self._load_faq()
        print(f"✅ Loaded {len(self.faq_data)} FAQ items")
        
        # Initialize optimized components
        self.sessions = SessionManager()
        self.search = OptimizedSearch(self.faq_data)
        self.response_builder = OptimizedResponseBuilder()
        
        print("✅ All systems optimized and ready")
    
    def _load_faq(self):
        """Efficient FAQ loading"""
        try:
            with open(FAQ_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception as e:
            print(f"❌ Error loading FAQ: {e}")
            return []
    
    def process(self, user_input, session_id):
        """Optimized processing pipeline"""
        start_time = time.time()
        user_input_lower = user_input.lower().strip()
        
        # 1. Quick safety checks first
        safety_response = self._check_safety(user_input_lower)
        if safety_response:
            return safety_response
        
        # 2. Handle special intents
        special_response = self._handle_special_intents(user_input_lower)
        if special_response:
            return special_response
        
        # 3. NLP analysis
        nlp_analysis = SmartNLP.analyze_query(user_input)
        medical_entities = AdvancedMedicalNLP.extract_medical_entities(user_input)
        
        # 4. Get conversation context
        context = self.sessions.get_context(session_id)
        
        # 5. Intelligent search
        matches = self.search.intelligent_search(user_input, nlp_analysis)
        
        # 6. Build response
        response = self.response_builder.build_response(
            user_input, matches, context, medical_entities
        )
        
        # 7. Update session
        self.sessions.update_session(
            session_id, user_input, response, nlp_analysis['medical_context']
        )
        
        processing_time = time.time() - start_time
        print(f"⏱️ Processing: {processing_time:.2f}s")
        
        return response
    
    def _check_safety(self, user_input):
        """Optimized safety checks"""
        # Critical emergencies (immediate danger)
        if any(emergency in user_input for emergency in CRITICAL_EMERGENCIES):
            return "🚨 MEDICAL EMERGENCY: Please call emergency services (911/112) immediately! This requires urgent medical attention."
        
        # Crisis situations (mental health)
        if any(crisis in user_input for crisis in CRISIS_KEYWORDS):
            return "🚨 CRISIS SUPPORT: Please contact these resources now: National Suicide Prevention Lifeline: 988, Crisis Text Line: Text HOME to 741741, Emergency Services: 911"
        
        # Urgent conditions (see doctor soon)
        if any(urgent in user_input for urgent in URGENT_CONDITIONS):
            return "🚨 URGENT: Please consult a healthcare provider immediately or visit urgent care. This requires prompt medical evaluation."
        
        return None
    
    def _handle_special_intents(self, user_input):
        """Handle special conversation cases"""
        # Only trigger on actual greetings
        if user_input in ['hello', 'hi', 'hey']:
            return "Hello! I'm your women's health assistant. I can help with PCOD, PCOS, menstrual health, hormones, and related topics. What would you like to know?"
        
        # Thanks handling
        if any(thanks in user_input for thanks in ['thank you', 'thanks']):
            return "You're welcome! I'm glad I could help. Is there anything else you'd like to know?"
        
        # Goodbye
        if any(bye in user_input for bye in ['bye', 'goodbye', 'exit', 'quit']):
            return "Thank you for chatting! Remember to consult healthcare professionals for personal medical advice. Take care! 💖"
        
        return None

# ================================
# TESTING UTILITIES
# ================================

def run_quick_tests():
    """Quick functionality tests"""
    print("🧪 RUNNING QUICK TESTS")
    print("=" * 50)
    
    bot = OptimizedMEBot()
    test_session = "test_" + str(int(time.time()))
    
    test_cases = [
        ("hello", "greeting"),
        ("I have chest pain", "emergency"),
        ("What is PCOD?", "medical"),
        ("Thank you", "thanks"),
        ("How to manage PCOS symptoms?", "complex"),
    ]
    
    for prompt, expected_type in test_cases:
        print(f"\n📝 Test: {prompt}")
        response = bot.process(prompt, test_session)
        print(f"Response: {response[:100]}...")
        
        # Basic validation
        if response and len(response) > 10:
            print("✅ PASS")
        else:
            print("❌ FAIL")
    
    print("\n🎯 Quick tests completed!")

# ================================
# MAIN CLI INTERFACE
# ================================

def run_optimized_cli():
    """Run the optimized bot"""
    bot = OptimizedMEBot()
    
    print(f"\n{BOT_NAME} - Optimized")
    print("=" * 50)
    print("✨ Fast & Accurate Medical AI")
    print("✨ Emergency Detection")
    print("✨ Context Awareness") 
    print("✨ Semantic Understanding")
    print("=" * 50)
    
    session_id = uuid.uuid4().hex[:8]
    print(f"Session: {session_id}")
    print("\nHello! I'm your optimized women's health assistant.")
    print("Type 'test' to run quick tests")
    print("Type 'bye' to exit")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            
            if user_input.lower() == 'test':
                run_quick_tests()
                continue
                
            if user_input.lower() in {'exit', 'quit', 'bye'}:
                print(f"{BOT_NAME}: Goodbye! 💖")
                break
            
            start_time = time.time()
            response = bot.process(user_input, session_id)
            processing_time = time.time() - start_time
            
            print(f"{BOT_NAME}: {response}")
            print(f"[Processing: {processing_time:.2f}s]")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print(f"\n{BOT_NAME}: Session ended. Take care!")
            break
        except Exception as e:
            print(f"{BOT_NAME}: I encountered an issue. Please try again.")
            print(f"Error: {e}")

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        run_quick_tests()
    else:
        run_optimized_cli()
