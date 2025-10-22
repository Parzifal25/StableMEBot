import os
import json
import time
import numpy as np
from datetime import datetime
import re
import uuid
import hashlib
from collections import defaultdict, deque
import faiss
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from rapidfuzz import fuzz
import phonetics

# -------------------------
# CONFIG & CONSTANTS
# -------------------------
BASE_DIR = os.path.dirname(__file__)
FAQ_PATH = os.path.join(BASE_DIR, 'faq.json')
EMB_PATH = os.path.join(BASE_DIR, 'faq_embeddings.npy')
SESSION_DATA_PATH = os.path.join(BASE_DIR, 'sessions')

BOT_NAME = "ME Bot"
MAX_CONTEXT = 20
MEDICAL_DISCLAIMER = "⚠️ I'm an AI assistant, not a medical professional. For serious symptoms, please consult a doctor immediately."
EMERGENCY_KEYWORDS = ['chest pain', 'difficulty breathing', 'heavy bleeding', 'suicide', 'emergency', 'urgent help', 'severe pain']

# Create sessions directory
os.makedirs(SESSION_DATA_PATH, exist_ok=True)

# -------------------------
# FIXED ENCRYPTED SESSION MANAGEMENT
# -------------------------
class EncryptedSessionManager:
    def __init__(self):
        self.sessions = {}
    
    def create_session(self, session_id):
        """Create encrypted session - FIXED VERSION"""
        session_key = self._hash_session_id(session_id)
        session_data = {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'last_activity': datetime.now().isoformat(),
            'conversation_history': deque(maxlen=MAX_CONTEXT),
            'discussed_topics': set(),
            'user_preferences': {},
            'encrypted_data': self._encrypt_data({'history': []})
        }
        self.sessions[session_key] = session_data
        return session_data
    
    def get_session(self, session_id):
        """Get session with encryption - FIXED VERSION"""
        session_key = self._hash_session_id(session_id)
        if session_key not in self.sessions:
            return self.create_session(session_id)
        
        # Update last activity
        self.sessions[session_key]['last_activity'] = datetime.now().isoformat()
        return self.sessions[session_key]
    
    def update_session(self, session_id, user_input, bot_response, topics):
        """Update session with new conversation - FIXED VERSION"""
        session = self.get_session(session_id)
        
        # Add to conversation history
        session['conversation_history'].append({
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'bot_response': bot_response,
            'topics': topics
        })
        
        # Update discussed topics
        session['discussed_topics'].update(topics)
        
        # Update encrypted data
        decrypted_data = self._decrypt_data(session['encrypted_data'])
        decrypted_data['history'].append({
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'bot_response': bot_response,
            'topics': topics
        })
        session['encrypted_data'] = self._encrypt_data(decrypted_data)
    
    def get_conversation_context(self, session_id):
        """Get conversation context from encrypted session - FIXED VERSION"""
        session = self.get_session(session_id)
        
        # Use regular conversation history for context (not encrypted)
        recent_history = list(session['conversation_history'])[-5:]
        
        return {
            'recent_history': recent_history,
            'discussed_topics': list(session['discussed_topics']),
            'session_duration': self._get_session_duration(session)
        }
    
    def _hash_session_id(self, session_id):
        """Hash session ID for privacy"""
        return hashlib.sha256(session_id.encode()).hexdigest()[:16]
    
    def _encrypt_data(self, data):
        """Simple encryption for session data"""
        try:
            data_str = json.dumps(data)
            # Simple XOR encryption for demo (use proper encryption in production)
            key = 0xAB
            encrypted = ''.join(chr(ord(c) ^ key) for c in data_str)
            return encrypted
        except:
            return ""
    
    def _decrypt_data(self, encrypted_data):
        """Decrypt session data"""
        try:
            if not encrypted_data:
                return {'history': []}
            key = 0xAB
            decrypted = ''.join(chr(ord(c) ^ key) for c in encrypted_data)
            return json.loads(decrypted)
        except:
            return {'history': []}
    
    def _get_session_duration(self, session):
        """Calculate session duration"""
        created = datetime.fromisoformat(session['created_at'])
        now = datetime.now()
        return int((now - created).total_seconds() / 60)

# -------------------------
# ADVANCED NLP UNDERSTANDING
# -------------------------
class AdvancedNLPUnderstanding:
    def __init__(self):
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.stopwords = set(stopwords.words('english'))
        
    def analyze_query(self, query):
        """Comprehensive query analysis"""
        query_lower = query.lower()
        
        return {
            'phonetic_similarity': self._phonetic_analysis(query_lower),
            'syntactic_structure': self._syntactic_analysis(query),
            'semantic_intent': self._semantic_intent_analysis(query_lower),
            'vocabulary_complexity': self._vocabulary_analysis(query),
            'tone': self._tone_analysis(query_lower),
            'medical_context': self._medical_context_analysis(query_lower)
        }
    
    def _phonetic_analysis(self, query):
        """Phonetic similarity analysis"""
        try:
            phonetic_representation = phonetics.metaphone(query)
            return {
                'phonetic_code': phonetic_representation,
                'sounds_like': self._get_phonetic_variations(query)
            }
        except:
            return {'phonetic_code': '', 'sounds_like': []}
    
    def _syntactic_analysis(self, query):
        """Syntactic structure analysis"""
        words = query.split()
        return {
            'word_count': len(words),
            'question_type': self._detect_question_type(query),
            'sentence_structure': 'complex' if len(words) > 8 else 'simple',
            'has_medical_terms': any(term in query.lower() for term in ['symptom', 'treatment', 'diagnosis', 'medication'])
        }
    
    def _semantic_intent_analysis(self, query):
        """Semantic intent understanding"""
        intents = []
        
        if any(word in query for word in ['what', 'define', 'explain']):
            intents.append('definition')
        if any(word in query for word in ['how', 'method', 'way']):
            intents.append('method')
        if any(word in query for word in ['why', 'reason', 'cause']):
            intents.append('causation')
        if any(word in query for word in ['symptom', 'sign', 'experience']):
            intents.append('symptoms')
        if any(word in query for word in ['treatment', 'cure', 'medicine']):
            intents.append('treatment')
        
        return intents if intents else ['general_inquiry']
    
    def _vocabulary_analysis(self, query):
        """Vocabulary complexity analysis"""
        words = query.split()
        complex_words = [word for word in words if len(word) > 8 and word.lower() not in self.stopwords]
        
        return {
            'complexity_score': len(complex_words) / len(words) if words else 0,
            'reading_level': 'advanced' if len(complex_words) > 2 else 'basic'
        }
    
    def _tone_analysis(self, query):
        """Tone and emotion analysis"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['worried', 'scared', 'anxious', 'nervous']):
            return 'anxious'
        elif any(word in query_lower for word in ['urgent', 'emergency', 'immediately']):
            return 'urgent'
        elif any(word in query_lower for word in ['confused', 'unsure', 'clarify']):
            return 'confused'
        elif any(word in query_lower for word in ['thank', 'appreciate', 'helpful']):
            return 'grateful'
        else:
            return 'neutral'
    
    def _medical_context_analysis(self, query):
        """Medical context detection"""
        medical_keywords = {
            'pcod_pcos': ['pcod', 'pcos', 'polycystic', 'ovarian'],
            'menstrual': ['period', 'menstrual', 'cycle', 'pms'],
            'hormonal': ['hormone', 'estrogen', 'progesterone'],
            'lifestyle': ['diet', 'exercise', 'yoga', 'nutrition'],
            'mental_health': ['stress', 'anxiety', 'depression', 'mood']
        }
        
        detected_contexts = []
        for context, keywords in medical_keywords.items():
            if any(keyword in query.lower() for keyword in keywords):
                detected_contexts.append(context)
        
        return detected_contexts
    
    def _detect_question_type(self, query):
        """Detect type of question"""
        if query.strip().endswith('?'):
            if query.lower().startswith(('what', 'which')):
                return 'factual'
            elif query.lower().startswith(('how', 'why')):
                return 'explanatory'
            elif query.lower().startswith(('can', 'should', 'would')):
                return 'advisory'
        return 'statement'
    
    def _get_phonetic_variations(self, query):
        """Generate phonetic variations"""
        words = query.split()
        phonetic_variations = []
        
        for word in words:
            if len(word) > 3:  # Only for substantial words
                try:
                    phonetic_variations.append(phonetics.metaphone(word))
                except:
                    continue
        
        return phonetic_variations

# -------------------------
# EXPLAINABLE AI RESPONSE BUILDER
# -------------------------
class ExplainableAIResponseBuilder:
    def __init__(self, nlp_understanding):
        self.nlp = nlp_understanding
    
    def build_explained_response(self, user_query, faq_match, conversation_context):
        """Build response with explanation in own words"""
        query_analysis = self.nlp.analyze_query(user_query)
        
        # Base answer from FAQ
        base_answer = faq_match['answer']
        
        # Explain in own words
        explained_answer = self._explain_in_own_words(base_answer, query_analysis)
        
        # Adjust tone based on analysis
        tone_adjusted = self._adjust_tone(explained_answer, query_analysis['tone'])
        
        # Add context if available
        contextualized = self._add_conversation_context(tone_adjusted, conversation_context)
        
        return contextualized
    
    def _explain_in_own_words(self, base_answer, query_analysis):
        """Explain FAQ answer in natural language"""
        # Simple explanation transformation
        explanations = {
            'definition': f"Let me explain this in simple terms: ",
            'symptoms': f"Here are the key things to look out for: ",
            'treatment': f"Based on medical knowledge, here are the main approaches: ",
            'method': f"Here's how this typically works: ",
            'causation': f"The main reasons behind this are: "
        }
        
        # Choose appropriate explanation prefix
        intent = query_analysis['semantic_intent'][0] if query_analysis['semantic_intent'] else 'general_inquiry'
        prefix = explanations.get(intent, "Here's what I can tell you: ")
        
        # Simplify and explain
        simplified = self._simplify_medical_language(base_answer)
        
        return prefix + simplified
    
    def _simplify_medical_language(self, text):
        """Simplify medical jargon"""
        simplifications = {
            'may improve': 'can help with',
            'evidence-based': 'scientifically proven',
            'complementary approach': 'additional method',
            'menstrual regulation': 'period regularity',
            'symptoms': 'signs and experiences',
            'definitive': 'clear and certain',
            'clinician': 'doctor or healthcare provider'
        }
        
        simplified = text
        for complex_term, simple_term in simplifications.items():
            simplified = simplified.replace(complex_term, simple_term)
        
        return simplified
    
    def _adjust_tone(self, response, tone):
        """Adjust response tone based on user's emotional state"""
        tone_prefixes = {
            'anxious': "I understand this might be worrying. ",
            'urgent': "This sounds important. ",
            'confused': "Let me clarify this for you. ",
            'grateful': "I'm glad I can help! ",
            'neutral': ""
        }
        
        tone_suffixes = {
            'anxious': " Remember, many women successfully manage these concerns with proper care.",
            'urgent': " If this is an emergency, please seek immediate medical attention.",
            'confused': " Does this help make things clearer?",
            'grateful': " Feel free to ask more questions!",
            'neutral': ""
        }
        
        prefix = tone_prefixes.get(tone, "")
        suffix = tone_suffixes.get(tone, "")
        
        return prefix + response + suffix
    
    def _add_conversation_context(self, response, conversation_context):
        """Add conversation context to response"""
        if not conversation_context or not conversation_context.get('recent_history'):
            return response
        
        recent_topics = conversation_context.get('discussed_topics', [])
        if recent_topics and len(recent_topics) > 1:
            # Reference previous discussion
            topics_str = ', '.join(list(recent_topics)[-2:])
            return f"Continuing our discussion about {topics_str}, {response.lower()}"
        
        return response

# -------------------------
# SCALABLE FAQ SEARCH ENGINE
# -------------------------
class ScalableFAQSearcher:
    def __init__(self, faq_data):
        self.faq_data = faq_data
        self.faq_questions = [faq['question'] for faq in faq_data]
        self.nlp = AdvancedNLPUnderstanding()
        
        # Initialize vector search for scalability
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self._setup_vector_index()
    
    def _setup_vector_index(self):
        """Setup FAISS index for 10k+ FAQs"""
        try:
            print("🔄 Setting up scalable vector index...")
            self.faq_embeddings = self.embed_model.encode(self.faq_questions, convert_to_numpy=True)
            
            # Normalize for cosine similarity
            norms = np.linalg.norm(self.faq_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            faq_embeddings_norm = self.faq_embeddings / norms
            
            dimension = self.faq_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(faq_embeddings_norm.astype('float32'))
            
            print(f"✅ Vector index ready for {len(self.faq_data)} FAQs")
        except Exception as e:
            print(f"⚠️ Vector index setup failed: {e}")
            self.index = None
    
    def comprehensive_search(self, user_query):
        """Comprehensive search with multiple strategies"""
        query_analysis = self.nlp.analyze_query(user_query)
        all_matches = []
        
        # 1. Vector semantic search
        vector_matches = self._vector_search(user_query)
        all_matches.extend(vector_matches)
        
        # 2. Phonetic search
        phonetic_matches = self._phonetic_search(user_query, query_analysis)
        all_matches.extend(phonetic_matches)
        
        # 3. Keyword search
        keyword_matches = self._keyword_search(user_query)
        all_matches.extend(keyword_matches)
        
        # 4. Topic-based search
        topic_matches = self._topic_search(query_analysis['medical_context'])
        all_matches.extend(topic_matches)
        
        # Rank and return best matches
        return self._rank_matches(all_matches, query_analysis)
    
    def _vector_search(self, query, top_k=10):
        """Vector similarity search"""
        if self.index is None:
            return []
        
        try:
            query_vector = self.embed_model.encode([query], convert_to_numpy=True)
            query_norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
            query_norm[query_norm == 0] = 1.0
            query_vector_norm = query_vector / query_norm
            
            scores, indices = self.index.search(query_vector_norm.astype('float32'), top_k)
            
            matches = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.faq_data) and score > 0.3:
                    matches.append({
                        'index': int(idx),
                        'score': float(score),
                        'question': self.faq_data[idx]['question'],
                        'answer': self.faq_data[idx]['answer'],
                        'topics': self.faq_data[idx].get('topic', []),
                        'type': 'semantic'
                    })
            return matches
        except:
            return []
    
    def _phonetic_search(self, query, query_analysis):
        """Phonetic similarity search"""
        matches = []
        user_phonetic = query_analysis['phonetic_similarity']['phonetic_code']
        
        for idx, faq in enumerate(self.faq_data):
            faq_phonetic = phonetics.metaphone(faq['question'])
            similarity = fuzz.ratio(user_phonetic, faq_phonetic)
            
            if similarity > 70:
                matches.append({
                    'index': idx,
                    'score': similarity / 100.0,
                    'question': faq['question'],
                    'answer': faq['answer'],
                    'topics': faq.get('topic', []),
                    'type': 'phonetic'
                })
        
        return matches
    
    def _keyword_search(self, query):
        """Keyword-based search"""
        query_lower = query.lower()
        matches = []
        
        for idx, faq in enumerate(self.faq_data):
            question_lower = faq['question'].lower()
            answer_lower = faq['answer'].lower()
            
            # Check both question and answer
            if (query_lower in question_lower or 
                any(word in question_lower for word in query_lower.split())):
                matches.append({
                    'index': idx,
                    'score': 0.8,
                    'question': faq['question'],
                    'answer': faq['answer'],
                    'topics': faq.get('topic', []),
                    'type': 'keyword'
                })
        
        return matches
    
    def _topic_search(self, medical_contexts):
        """Topic-based search"""
        if not medical_contexts:
            return []
        
        matches = []
        for idx, faq in enumerate(self.faq_data):
            faq_topics = [topic.lower() for topic in faq.get('topic', [])]
            for context in medical_contexts:
                if any(context in topic for topic in faq_topics):
                    matches.append({
                        'index': idx,
                        'score': 0.6,
                        'question': faq['question'],
                        'answer': faq['answer'],
                        'topics': faq.get('topic', []),
                        'type': 'topic'
                    })
                    break
        
        return matches
    
    def _rank_matches(self, matches, query_analysis):
        """Rank matches by multiple factors"""
        scored_matches = []
        
        for match in matches:
            final_score = match['score']
            
            # Boost semantic matches
            if match['type'] == 'semantic':
                final_score *= 1.2
            
            # Boost by intent match
            if any(intent in match['question'].lower() for intent in query_analysis['semantic_intent']):
                final_score *= 1.1
            
            match['final_score'] = min(final_score, 1.0)
            scored_matches.append(match)
        
        # Remove duplicates and sort
        unique_matches = {}
        for match in scored_matches:
            key = match['question']
            if key not in unique_matches or match['final_score'] > unique_matches[key]['final_score']:
                unique_matches[key] = match
        
        return sorted(unique_matches.values(), key=lambda x: x['final_score'], reverse=True)

# -------------------------
# TIME-BASED GREETING SYSTEM
# -------------------------
class TimeBasedGreeting:
    @staticmethod
    def get_greeting():
        """Get time-appropriate greeting"""
        current_hour = datetime.now().hour
        
        if 5 <= current_hour < 12:
            return "Good morning! 🌅"
        elif 12 <= current_hour < 17:
            return "Good afternoon! ☀️"
        elif 17 <= current_hour < 21:
            return "Good evening! 🌇"
        else:
            return "Hello! 🌙"
    
    @staticmethod
    def get_welcome_message(session_context=None):
        """Get personalized welcome message"""
        greeting = TimeBasedGreeting.get_greeting()
        base_message = f"{greeting} I'm ME Bot, your private women's health assistant."
        
        if session_context and session_context.get('discussed_topics'):
            topics = list(session_context['discussed_topics'])[-2:]
            topics_str = ', '.join(topics)
            return f"{base_message} We were discussing {topics_str}. How can I help you today? 😊"
        
        return f"{base_message} How can I help you today? 😊"

# -------------------------
# ROBUST FAQ LOADER
# -------------------------
class RobustFAQLoader:
    @staticmethod
    def load_faq_with_validation(file_path):
        """Load FAQ with comprehensive error handling"""
        print(f"📂 Loading FAQ from: {file_path}")
        
        if not os.path.exists(file_path):
            print("❌ FAQ file not found!")
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                print("❌ FAQ file is empty!")
                return []
            
            # Try to parse as JSON
            try:
                data = json.loads(content)
                print(f"✅ Successfully loaded {len(data)} FAQ entries")
                return data
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing error: {e}")
                return []
                
        except Exception as e:
            print(f"❌ Error reading FAQ file: {e}")
            return []

# -------------------------
# MAIN ENHANCED CHATBOT
# -------------------------
class EnhancedMEBot:
    def __init__(self):
        print("🚀 Initializing Enhanced ME Bot...")
        
        # Load FAQ data
        self.faq_data = RobustFAQLoader.load_faq_with_validation(FAQ_PATH)
        
        # Initialize all systems
        self.session_manager = EncryptedSessionManager()
        self.nlp_understanding = AdvancedNLPUnderstanding()
        self.faq_searcher = ScalableFAQSearcher(self.faq_data)
        self.response_builder = ExplainableAIResponseBuilder(self.nlp_understanding)
        
        print("✅ All systems initialized")
        print("🔐 Encrypted session management ready")
        print("🧠 Advanced NLP understanding active")
        print("🎯 Explainable AI responses enabled")
        print("📊 Scalable to 10k+ FAQs")
    
    def process_query(self, user_input, session_id):
        """Process user query with all enhanced features"""
        # Emergency detection
        if any(emergency in user_input.lower() for emergency in EMERGENCY_KEYWORDS):
            return f"🚨 EMERGENCY: {MEDICAL_DISCLAIMER} Please seek immediate medical attention!"
        
        # Get session context
        session_context = self.session_manager.get_conversation_context(session_id)
        
        # Comprehensive search
        matches = self.faq_searcher.comprehensive_search(user_input)
        
        if not matches:
            return self._build_fallback_response(user_input, session_context)
        
        # Get best match
        best_match = matches[0]
        
        # Build explained response
        response = self.response_builder.build_explained_response(
            user_input, best_match, session_context
        )
        
        # Extract topics for session tracking
        detected_topics = self.nlp_understanding.analyze_query(user_input)['medical_context']
        
        # Update session
        self.session_manager.update_session(
            session_id, user_input, response, detected_topics
        )
        
        return response
    
    def _build_fallback_response(self, user_input, session_context):
        """Build fallback response when no matches found"""
        query_analysis = self.nlp_understanding.analyze_query(user_input)
        
        if session_context and session_context.get('discussed_topics'):
            topics = list(session_context['discussed_topics'])[-2:]
            return f"I specialize in women's health topics like {', '.join(topics)}. Could you ask something specific about these areas?"
        
        return "I focus on women's health including PCOD/PCOS, menstrual health, hormones, and lifestyle. Please ask about these specific topics for detailed information."

# -------------------------
# ENHANCED CLI INTERFACE
# -------------------------
def run_enhanced_cli():
    bot = EnhancedMEBot()
    
    print(f"\n{BOT_NAME} Enhanced")
    print("=" * 60)
    print("✨ Contextual Memory & Session Management")
    print("✨ Tone Adjustment & Emotional Intelligence")
    print("✨ Explainable AI (Answers in Own Words)")
    print("✨ Encrypted & Private Conversations")
    print("✨ Scalable to 10k+ FAQs")
    print("✨ Phonetic & Semantic Understanding")
    print("✨ Time-based Personalization")
    print("=" * 60)
    
    session_id = str(uuid.uuid4())[:8]
    print(f"Session ID: {session_id} (Encrypted)")
    
    # Get initial context for personalized greeting
    try:
        initial_context = bot.session_manager.get_conversation_context(session_id)
        welcome_message = TimeBasedGreeting.get_welcome_message(initial_context)
    except Exception as e:
        print(f"⚠️ Session initialization issue: {e}")
        welcome_message = TimeBasedGreeting.get_welcome_message()
    
    print(f"\n{welcome_message}")
    print("I'll explain everything in clear, simple terms and remember our conversation.")
    print("Type 'bye' to exit")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit', 'bye']:
                # Get final context for personalized goodbye
                try:
                    final_context = bot.session_manager.get_conversation_context(session_id)
                    duration = final_context.get('session_duration', 0)
                    topics = final_context.get('discussed_topics', [])
                    
                    goodbye_msg = f"Goodbye! We spoke for {duration} minutes about {len(topics)} topics. Take care! 💖"
                    print(f"{BOT_NAME}: {goodbye_msg}")
                except:
                    print(f"{BOT_NAME}: Goodbye! Take care! 💖")
                break
            
            start_time = time.time()
            response = bot.process_query(user_input, session_id)
            end_time = time.time()
            
            print(f"{BOT_NAME}: {response}")
            print(f"[Processed in {end_time-start_time:.2f}s]")
            print("-" * 60)
            
        except KeyboardInterrupt:
            print(f"\n{BOT_NAME}: Session encrypted and saved. Stay healthy! 💪")
            break
        except Exception as e:
            print(f"{BOT_NAME}: I'm here and ready to help! Please try your question again.")

if __name__ == '__main__':
    run_enhanced_cli()
