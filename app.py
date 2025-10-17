import os
import json
import time
import numpy as np
from datetime import datetime
import re
import uuid
from collections import defaultdict, deque
import faiss
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords

# -------------------------
# CONFIG & CONSTANTS
# -------------------------
BASE_DIR = os.path.dirname(__file__)
FAQ_PATH = os.path.join(BASE_DIR, 'faq.json')
EMB_PATH = os.path.join(BASE_DIR, 'faq_embeddings.npy')

BOT_NAME = "ME Bot"
SEM_THRESHOLD_HIGH = 0.75
SEM_THRESHOLD_MEDIUM = 0.55
MAX_CONTEXT = 15
MEDICAL_DISCLAIMER = "⚠️ I'm an AI assistant, not a medical professional. For serious symptoms, please consult a doctor immediately."
EMERGENCY_KEYWORDS = ['chest pain', 'difficulty breathing', 'heavy bleeding', 'suicide', 'emergency', 'urgent help', 'severe pain']

# Medical context keywords that require doctor consultation
MEDICAL_CONSULT_KEYWORDS = [
    'pain', 'bleeding', 'fever', 'lump', 'swelling', 'infection', 'abnormal',
    'discharge', 'irregular', 'severe', 'chronic', 'persistent', 'worsening'
]

# -------------------------
# NLP INITIALIZATION
# -------------------------
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

STOPWORDS = set(stopwords.words('english'))
EMBED_MODEL_NAME = 'all-MiniLM-L6-v2'
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# -------------------------
# TIME-BASED GREETING SYSTEM
# -------------------------
class TimeBasedGreeting:
    @staticmethod
    def get_greeting():
        """Get appropriate greeting based on time of day"""
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
    def get_welcome_message():
        """Get complete welcome message with time-based greeting"""
        greeting = TimeBasedGreeting.get_greeting()
        return f"{greeting} I'm ME Bot, your women's health assistant. How can I help you today? 😊"

# -------------------------
# ENHANCED CONTEXT MEMORY
# -------------------------
class EnhancedContextMemory:
    def __init__(self, session_id):
        self.session_id = session_id
        self.conversation_history = deque(maxlen=MAX_CONTEXT)
        self.discussed_topics = deque(maxlen=8)
        self.current_topic = None
        self.user_interests = defaultdict(int)
        self.conversation_start_time = datetime.now()
        self.last_interaction_time = datetime.now()
        
    def add_exchange(self, user_input, bot_response, detected_topics=None, user_intent=None):
        """Add conversation exchange with enhanced context tracking"""
        exchange = {
            'user': user_input,
            'bot': bot_response,
            'timestamp': datetime.now().isoformat(),
            'topics': detected_topics or [],
            'intent': user_intent,
            'entities': self.extract_medical_entities(user_input)
        }
        
        self.conversation_history.append(exchange)
        self.last_interaction_time = datetime.now()
        
        # Update topic tracking
        if detected_topics:
            self.current_topic = detected_topics[0] if detected_topics else None
            for topic in detected_topics:
                self.discussed_topics.append(topic)
                self.user_interests[topic] += 1
        
        # Update user interests based on entities
        for entity in exchange['entities']:
            self.user_interests[entity] += 1
    
    def extract_medical_entities(self, text):
        """Extract medical entities from text with enhanced detection"""
        text_lower = text.lower()
        entities = []
        
        medical_entities = {
            'pcod': ['pcod', 'polycystic ovarian disease'],
            'pcos': ['pcos', 'polycystic ovary syndrome'],
            'period': ['period', 'menstrual', 'menstruation', 'pms'],
            'hormone': ['hormone', 'hormonal', 'estrogen', 'progesterone', 'testosterone'],
            'pregnancy': ['pregnancy', 'pregnant', 'fertility', 'conceive', 'ovulation'],
            'diet': ['diet', 'food', 'nutrition', 'eat', 'meal', 'weight'],
            'exercise': ['exercise', 'workout', 'yoga', 'physical activity'],
            'symptom': ['symptom', 'pain', 'cramp', 'bleeding', 'discharge', 'headache'],
            'stress': ['stress', 'anxiety', 'depression', 'mental', 'mood'],
            'treatment': ['treatment', 'medicine', 'medication', 'therapy', 'cure']
        }
        
        for entity, keywords in medical_entities.items():
            if any(keyword in text_lower for keyword in keywords):
                entities.append(entity)
        
        return list(set(entities))
    
    def get_conversation_context(self):
        """Get comprehensive conversation context"""
        recent_history = list(self.conversation_history)[-4:]  # Last 4 exchanges
        
        # Get frequent interests
        frequent_interests = sorted(self.user_interests.items(), 
                                  key=lambda x: x[1], reverse=True)[:5]
        
        # Calculate conversation duration
        conversation_duration = datetime.now() - self.conversation_start_time
        
        return {
            'recent_history': recent_history,
            'current_topic': self.current_topic,
            'recent_topics': list(self.discussed_topics)[-4:],
            'frequent_interests': [interest for interest, count in frequent_interests],
            'conversation_duration_minutes': int(conversation_duration.total_seconds() / 60),
            'exchange_count': len(self.conversation_history)
        }
    
    def is_follow_up_question(self, current_query, new_intent):
        """Enhanced follow-up detection"""
        if not self.conversation_history:
            return False
        
        last_exchange = self.conversation_history[-1]
        last_user_input = last_exchange['user'].lower()
        current_query_lower = current_query.lower()
        
        # Follow-up indicators
        follow_up_indicators = [
            'what about', 'how about', 'and what', 'also', 'then',
            'explain', 'clarify', 'more about', 'tell me more',
            'so', 'and', 'next', 'another'
        ]
        
        # Check for follow-up words
        has_follow_up_word = any(indicator in current_query_lower for indicator in follow_up_indicators)
        
        # Check topic continuity
        current_entities = set(self.extract_medical_entities(current_query))
        last_entities = set(last_exchange['entities'])
        same_topic = len(current_entities & last_entities) > 0
        
        # Check intent continuity
        last_intent = last_exchange.get('intent')
        intent_continuity = last_intent == new_intent
        
        return has_follow_up_word or same_topic or intent_continuity or len(current_query.split()) <= 5

# -------------------------
# SMART INTENT DETECTION SYSTEM
# -------------------------
class SmartIntentDetection:
    def __init__(self):
        self.intent_patterns = {
            'definition': {
                'patterns': ['what is', 'what are', 'tell me about', 'explain', 'define', 'meaning of'],
                'weight': 1.2
            },
            'symptoms': {
                'patterns': ['symptoms', 'signs', 'indicators', 'how to know', 'experience', 'feel', 'pain', 'cramp'],
                'weight': 1.3
            },
            'causes': {
                'patterns': ['causes', 'reasons', 'why', 'what causes', 'reason for', 'trigger', 'lead to'],
                'weight': 1.1
            },
            'treatment': {
                'patterns': ['treatment', 'cure', 'medicine', 'medication', 'how to treat', 'management', 'control', 'solution'],
                'weight': 1.4
            },
            'prevention': {
                'patterns': ['prevent', 'avoid', 'stop', 'reduce risk', 'lower risk', 'precaution', 'preventive'],
                'weight': 1.1
            },
            'lifestyle': {
                'patterns': ['diet', 'food', 'nutrition', 'exercise', 'yoga', 'workout', 'lifestyle', 'habit', 'sleep'],
                'weight': 1.0
            },
            'comparison': {
                'patterns': ['difference', 'compare', 'vs', 'versus', 'between', 'similar', 'different from'],
                'weight': 1.2
            },
            'severity': {
                'patterns': ['serious', 'dangerous', 'emergency', 'worsen', 'worse', 'severe', 'critical', 'bad', 'harmful'],
                'weight': 1.5
            },
            'advice': {
                'patterns': ['should i', 'can i', 'would you recommend', 'what should', 'how can i', 'is it safe'],
                'weight': 1.3
            }
        }
        
        self.topic_keywords = {
            'pcod_pcos': ['pcod', 'pcos', 'polycystic', 'ovarian', 'cyst', 'ovary'],
            'menstrual': ['period', 'menstrual', 'cycle', 'pms', 'cramp', 'bleeding', 'menstruation'],
            'hormonal': ['hormone', 'hormonal', 'estrogen', 'progesterone', 'testosterone', 'endocrine'],
            'fertility': ['fertility', 'pregnant', 'pregnancy', 'conceive', 'ovulation', 'infertility'],
            'stress_mental': ['stress', 'anxiety', 'depression', 'mental', 'mood', 'emotional', 'psychological'],
            'diet_nutrition': ['diet', 'food', 'nutrition', 'eat', 'meal', 'weight', 'obesity', 'dietary'],
            'reproductive': ['reproductive', 'uterine', 'vaginal', 'cervical', 'fallopian', 'endometriosis']
        }
    
    def analyze_intent(self, query):
        """Advanced intent analysis with confidence scoring"""
        query_lower = query.lower()
        
        intent_scores = defaultdict(float)
        detected_intents = []
        
        # Score each intent
        for intent, data in self.intent_patterns.items():
            patterns = data['patterns']
            weight = data['weight']
            
            for pattern in patterns:
                if pattern in query_lower:
                    intent_scores[intent] += weight
                    if intent not in detected_intents:
                        detected_intents.append(intent)
        
        # Detect topics
        detected_topics = []
        for topic, keywords in self.topic_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_topics.append(topic)
        
        # Determine primary intent
        primary_intent = 'general'
        confidence = 0.0
        
        if intent_scores:
            primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
            max_score = max(intent_scores.values())
            confidence = min(max_score / 2.0, 1.0)  # Normalize to 0-1
        
        # Detect question type
        question_type = 'factual'
        if any(word in query_lower for word in ['should', 'can i', 'would', 'could', 'recommend']):
            question_type = 'advice'
        elif any(word in query_lower for word in ['why', 'reason', 'cause']):
            question_type = 'explanatory'
        
        return {
            'primary_intent': primary_intent,
            'all_intents': detected_intents,
            'intent_scores': dict(intent_scores),
            'topics': detected_topics,
            'question_type': question_type,
            'confidence': confidence,
            'requires_doctor': self.requires_doctor_consultation(query_lower)
        }
    
    def requires_doctor_consultation(self, query_lower):
        """Check if query requires doctor consultation"""
        high_concern_indicators = [
            'severe pain', 'heavy bleeding', 'fever with', 'lump in', 'unusual discharge',
            'can\'t breathe', 'chest pain', 'sharp pain', 'continuous pain'
        ]
        
        if any(indicator in query_lower for indicator in high_concern_indicators):
            return True
        
        # Check for medical context keywords
        medical_words = sum(1 for word in MEDICAL_CONSULT_KEYWORDS if word in query_lower)
        return medical_words >= 2
    
    def expand_query_variations(self, query):
        """Generate intelligent query variations"""
        query_lower = query.lower()
        variations = [query_lower]
        
        # Synonym-based expansions
        synonym_map = {
            'pcod': ['pcos', 'polycystic ovarian disease', 'polycystic ovary syndrome'],
            'pcos': ['pcod', 'polycystic ovary syndrome', 'polycystic ovarian disease'],
            'stress': ['anxiety', 'mental stress', 'psychological stress', 'pressure'],
            'period': ['menstrual cycle', 'menstruation', 'monthly cycle', 'menses'],
            'diet': ['nutrition', 'food', 'eating habits', 'meal plan'],
            'exercise': ['workout', 'physical activity', 'fitness', 'yoga']
        }
        
        for original, synonyms in synonym_map.items():
            if original in query_lower:
                for synonym in synonyms:
                    variations.append(query_lower.replace(original, synonym))
        
        # Intent-based expansions
        intent_analysis = self.analyze_intent(query)
        if intent_analysis['primary_intent'] == 'symptoms':
            variations.extend([f"symptoms of {query_lower}", f"signs of {query_lower}"])
        elif intent_analysis['primary_intent'] == 'treatment':
            variations.extend([f"treatment for {query_lower}", f"how to treat {query_lower}"])
        
        return list(set([v.strip() for v in variations if v and len(v) > 3]))

# -------------------------
# OPTIMIZED VECTOR SEARCH ENGINE
# -------------------------
class OptimizedVectorSearch:
    def __init__(self, faq_data):
        self.faq_data = faq_data
        self.faq_questions = [faq['question'] for faq in faq_data]
        self.faq_answers = [faq['answer'] for faq in faq_data]
        
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.faq_embeddings = None
        
        self.setup_vector_index()
    
    def setup_vector_index(self):
        """Create FAISS index for fast vector similarity search"""
        print("🔄 Building vector index for fast semantic search...")
        
        # Convert all FAQ questions to vectors
        self.faq_embeddings = self.embed_model.encode(
            self.faq_questions, 
            convert_to_numpy=True, 
            show_progress_bar=True,
            batch_size=64
        )
        
        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(self.faq_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.faq_embeddings_normalized = self.faq_embeddings / norms
        
        # Create FAISS index
        dimension = self.faq_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        
        # Add vectors to index
        self.index.add(self.faq_embeddings_normalized.astype('float32'))
        
        print(f"✅ Vector index built with {len(self.faq_questions)} FAQs")
        print(f"📊 Vector dimensions: {dimension}")
    
    def fast_semantic_search(self, query, top_k=10, similarity_threshold=0.3):
        """Lightning-fast vector similarity search using FAISS"""
        # Convert query to vector
        query_vector = self.embed_model.encode([query], convert_to_numpy=True)
        
        # Normalize query vector
        query_norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
        query_norm[query_norm == 0] = 1.0
        query_vector_normalized = query_vector / query_norm
        
        # FAISS search
        scores, indices = self.index.search(
            query_vector_normalized.astype('float32'), 
            top_k
        )
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.faq_questions) and score > similarity_threshold:
                results.append({
                    'index': int(idx),
                    'score': float(score),
                    'question': self.faq_questions[idx],
                    'answer': self.faq_answers[idx],
                    'type': 'vector_semantic'
                })
        
        return results

# -------------------------
# MULTI-STRATEGY FAQ MATCHER
# -------------------------
class MultiStrategyFAQMatcher:
    def __init__(self, faq_data):
        self.faq_data = faq_data
        self.vector_search = OptimizedVectorSearch(faq_data)
        self.intent_system = SmartIntentDetection()
    
    def find_best_match(self, user_query):
        """Intelligent matching combining multiple strategies"""
        intent_analysis = self.intent_system.analyze_intent(user_query)
        query_variations = self.intent_system.expand_query_variations(user_query)
        
        all_results = []
        
        # Strategy 1: Vector semantic search with all variations
        for query_var in query_variations:
            vector_results = self.vector_search.fast_semantic_search(query_var, top_k=8)
            all_results.extend(vector_results)
        
        # Strategy 2: Keyword reinforcement for high-confidence intents
        if intent_analysis['confidence'] > 0.6:
            keyword_results = self.keyword_reinforcement_search(user_query, intent_analysis)
            all_results.extend(keyword_results)
        
        # Strategy 3: Score and rank with intent boosting
        scored_results = self.score_with_intent_boost(all_results, intent_analysis, user_query)
        
        return scored_results[:5]  # Return top 5 matches for context
    
    def keyword_reinforcement_search(self, query, intent_analysis):
        """Keyword-based search to reinforce high-confidence matches"""
        query_lower = query.lower()
        matches = []
        
        for idx, faq_question in enumerate(self.faq_questions):
            faq_lower = faq_question.lower()
            
            # Calculate keyword overlap
            query_words = set(query_lower.split())
            faq_words = set(faq_lower.split())
            common_words = query_words & faq_words
            
            if len(common_words) >= 2:
                score = len(common_words) / max(len(query_words), len(faq_words))
                
                # Intent-based boosting
                if intent_analysis['primary_intent'] in faq_lower:
                    score += 0.2
                
                if score > 0.4:
                    matches.append({
                        'index': idx,
                        'score': score,
                        'question': faq_question,
                        'answer': self.faq_answers[idx],
                        'type': 'keyword_reinforcement'
                    })
        
        return matches
    
    def score_with_intent_boost(self, results, intent_analysis, original_query):
        """Boost scores based on intent and context matching"""
        for result in results:
            final_score = result['score']
            
            # Intent matching boost
            if intent_analysis['primary_intent'] != 'general':
                intent_words = self.intent_system.intent_patterns[intent_analysis['primary_intent']]['patterns']
                if any(word in result['question'].lower() for word in intent_words):
                    final_score += 0.15
            
            # Topic matching boost
            if intent_analysis['topics']:
                topic_boost = any(topic in result['question'].lower() 
                                for topic in intent_analysis['topics'])
                if topic_boost:
                    final_score += 0.1
            
            # Exact phrase bonus
            original_lower = original_query.lower()
            if any(phrase in result['question'].lower() for phrase in ['what is', 'how to', 'symptoms of', 'causes of']):
                if any(phrase in original_lower for phrase in ['what is', 'how to', 'symptoms', 'causes']):
                    final_score += 0.1
            
            result['final_score'] = min(final_score, 1.0)
        
        # Remove duplicates and sort
        unique_results = {}
        for result in results:
            key = result['question']
            if key not in unique_results or result['final_score'] > unique_results[key]['final_score']:
                unique_results[key] = result
        
        return sorted(unique_results.values(), key=lambda x: x['final_score'], reverse=True)

# -------------------------
# INTELLIGENT RESPONSE BUILDER
# -------------------------
class IntelligentResponseBuilder:
    def __init__(self, faq_matcher):
        self.faq_matcher = faq_matcher
        self.intent_system = SmartIntentDetection()
    
    def build_response(self, user_query, matches, conversation_context=None):
        """Build intelligent response with all enhancements"""
        # Handle casual conversation
        casual_response = self.handle_casual_conversation(user_query)
        if casual_response:
            return casual_response
        
        intent_analysis = self.intent_system.analyze_intent(user_query)
        
        # Doctor consultation warning for concerning queries
        if intent_analysis['requires_doctor']:
            doctor_warning = "🚨 Based on your description, I strongly recommend consulting a healthcare professional for proper evaluation. "
            if matches:
                return doctor_warning + f"Meanwhile, here's general information: {matches[0]['answer']}"
            else:
                return doctor_warning + "This sounds like it needs medical attention."
        
        if not matches:
            return self.build_contextual_fallback(user_query, intent_analysis, conversation_context)
        
        best_match = matches[0]
        
        if best_match['final_score'] >= SEM_THRESHOLD_HIGH:
            return self.enhance_high_confidence_match(best_match, user_query, intent_analysis, conversation_context)
        elif best_match['final_score'] >= SEM_THRESHOLD_MEDIUM:
            return self.build_combined_response(matches, user_query, intent_analysis, conversation_context)
        else:
            return self.build_cautious_response(best_match, user_query, intent_analysis, conversation_context)
    
    def handle_casual_conversation(self, user_query):
        """Handle casual conversation with time-based greetings"""
        query_lower = user_query.lower().strip()
        
        greeting_triggers = ['hello', 'hi', 'hey', 'hola', 'good morning', 'good afternoon', 'good evening']
        
        if any(trigger in query_lower for trigger in greeting_triggers):
            return TimeBasedGreeting.get_welcome_message()
        
        if query_lower in ['bye', 'goodbye', 'exit', 'quit']:
            return f"{BOT_NAME}: Goodbye! Take care of yourself! 💖 Remember to consult a doctor for any health concerns."
        
        if 'how are you' in query_lower:
            return f"{BOT_NAME}: I'm doing great, thank you! {TimeBasedGreeting.get_greeting()} Ready to help with your health questions. How are you feeling today? 🌸"
        
        if any(word in query_lower for word in ['thank', 'thanks', 'thank you']):
            return f"{BOT_NAME}: You're welcome! I'm happy I could help. 😊 Remember I'm here for any other women's health questions!"
        
        if any(word in query_lower for word in ['who are you', 'what are you', 'your name']):
            return f"{BOT_NAME}: I'm ME Bot, your dedicated women's health assistant! {TimeBasedGreeting.get_greeting()} I specialize in PCOD, PCOS, menstrual health, and related topics. How can I help you? 💫"
        
        return None
    
    def enhance_high_confidence_match(self, match, user_query, intent_analysis, conversation_context):
        """Enhance high-confidence matches with natural language and context"""
        base_answer = match['answer']
        
        # Context-aware enhancements
        context_prefix = ""
        if conversation_context and conversation_context.get('is_follow_up'):
            context_prefix = "To follow up on our previous discussion, "
        elif conversation_context and conversation_context.get('current_topic'):
            context_prefix = "Continuing with our topic, "
        
        # Intent-based natural enhancement
        if intent_analysis['primary_intent'] == 'definition':
            if not base_answer.startswith(('It is', 'This is', 'These are', 'In women')):
                enhanced = f"{context_prefix}In women's health, this refers to: {base_answer}"
            else:
                enhanced = context_prefix + base_answer
        
        elif intent_analysis['primary_intent'] == 'symptoms':
            if 'symptom' not in base_answer.lower() and 'sign' not in base_answer.lower():
                enhanced = f"{context_prefix}Common signs and symptoms include: {base_answer}"
            else:
                enhanced = context_prefix + base_answer
        
        elif intent_analysis['primary_intent'] == 'causes':
            if not any(word in base_answer.lower() for word in ['cause', 'due to', 'because', 'reason', 'factor']):
                enhanced = f"{context_prefix}This can be caused by several factors: {base_answer}"
            else:
                enhanced = context_prefix + base_answer
        
        elif intent_analysis['primary_intent'] == 'treatment':
            if not any(word in base_answer.lower() for word in ['treatment', 'manage', 'therapy', 'medication', 'approach']):
                enhanced = f"{context_prefix}Common management approaches include: {base_answer}"
            else:
                enhanced = context_prefix + base_answer
        
        elif intent_analysis['primary_intent'] == 'severity':
            enhanced = f"{context_prefix}Regarding severity: {base_answer}"
            if 'consult' not in base_answer.lower() and 'doctor' not in base_answer.lower():
                enhanced += " If symptoms are severe or persistent, please consult a healthcare provider."
        
        else:
            enhanced = context_prefix + base_answer
        
        return enhanced
    
    def build_combined_response(self, matches, user_query, intent_analysis, conversation_context):
        """Combine knowledge from multiple high-quality matches"""
        primary_answer = matches[0]['answer']
        
        # Find complementary information
        additional_info = []
        for match in matches[1:3]:
            if match['final_score'] > 0.5 and not self.is_similar_content(primary_answer, match['answer']):
                additional_info.append(match['answer'])
                if len(additional_info) >= 1:  # Limit to one additional piece
                    break
        
        if additional_info:
            return f"{primary_answer} Additionally, {additional_info[0]}"
        else:
            return primary_answer
    
    def build_cautious_response(self, match, user_query, intent_analysis, conversation_context):
        """Build careful response for low-confidence matches"""
        base_answer = match['answer']
        
        if intent_analysis['confidence'] < 0.3:
            prefix = "Based on general women's health knowledge, "
            if not base_answer.startswith(prefix):
                return prefix + base_answer.lower()
        
        return base_answer
    
    def build_contextual_fallback(self, user_query, intent_analysis, conversation_context):
        """Build intelligent fallback using context and intent"""
        # Use conversation context for better fallback
        if conversation_context:
            recent_topics = conversation_context.get('recent_topics', [])
            frequent_interests = conversation_context.get('frequent_interests', [])
            
            if recent_topics:
                topics_str = ', '.join([t.replace('_', ' ') for t in recent_topics[-2:]])
                return f"I notice we've been discussing {topics_str}. Could you ask something more specific about these areas?"
            
            if frequent_interests:
                interests_str = ', '.join([i.replace('_', ' ') for i in frequent_interests[:2]])
                return f"Based on our conversation, I can help with {interests_str}. What would you like to know specifically?"
        
        # Intent-based fallback
        if intent_analysis['topics']:
            topics_str = ', '.join([t.replace('_', ' ') for t in intent_analysis['topics']])
            return f"I specialize in {topics_str}. Could you rephrase your question or ask something specific about these topics?"
        
        # General fallback with safety reminder
        return "I focus on women's health topics like PCOD, PCOS, menstrual health, hormones, and lifestyle. " \
               "For specific medical concerns, please consult a healthcare provider. " \
               "What would you like to know about?"
    
    def is_similar_content(self, text1, text2):
        """Check if two texts contain very similar information"""
        words1 = set(text1.lower().split()[:10])
        words2 = set(text2.lower().split()[:10])
        return len(words1 & words2) >= 5

# -------------------------
# MAIN CHATBOT - MEbot v1.5 ENHANCED
# -------------------------
class MEBotV15Enhanced:
    def __init__(self):
        print("🚀 Initializing ME Bot v1.5 Enhanced...")
        
        # Load FAQ data
        self.faq_data = self.load_faq_data()
        print(f"✅ Loaded FAQ with {len(self.faq_data)} entries")
        
        # Initialize all intelligent systems
        self.faq_matcher = MultiStrategyFAQMatcher(self.faq_data)
        self.response_builder = IntelligentResponseBuilder(self.faq_matcher)
        self.intent_system = SmartIntentDetection()
        self.active_sessions = {}
        
        print("✅ All intelligent systems initialized")
        print("⚡ FAISS-powered vector search ready")
        print("🎯 Smart intent detection active")
        print("💾 Enhanced context memory enabled")
        print("✅ ME Bot v1.5 Enhanced initialized successfully!")
    
    def load_faq_data(self):
        """Load FAQ data from JSON file"""
        try:
            with open(FAQ_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"❌ Error loading FAQ: {e}")
            return []
    
    def get_session(self, session_id):
        """Get or create enhanced session"""
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = EnhancedContextMemory(session_id)
        return self.active_sessions[session_id]
    
    def detect_emergency(self, user_input):
        """Enhanced emergency detection"""
        user_input_lower = user_input.lower()
        return any(emergency in user_input_lower for emergency in EMERGENCY_KEYWORDS)
    
    def process_query(self, user_input, session_id="default"):
        """Process user query with all intelligent systems"""
        # Emergency detection
        if self.detect_emergency(user_input):
            return f"🚨 EMERGENCY: {MEDICAL_DISCLAIMER} Please contact emergency services or visit the nearest hospital immediately!"
        
        # Get session and context
        session = self.get_session(session_id)
        
        # Analyze intent
        intent_analysis = self.intent_system.analyze_intent(user_input)
        
        # Check if follow-up question
        is_follow_up = session.is_follow_up_question(user_input, intent_analysis['primary_intent'])
        
        # Find best FAQ matches
        matches = self.faq_matcher.find_best_match(user_input)
        
        # Get conversation context
        conversation_context = session.get_conversation_context()
        conversation_context['is_follow_up'] = is_follow_up
        
        # Build intelligent response
        response = self.response_builder.build_response(user_input, matches, conversation_context)
        
        # Update session with enhanced context
        session.add_exchange(user_input, response, 
                           intent_analysis['topics'], 
                           intent_analysis['primary_intent'])
        
        return response

# -------------------------
# CLI INTERFACE
# -------------------------
def run_cli():
    bot = MEBotV15Enhanced()
    
    print(f"\n{BOT_NAME} v1.5 Enhanced")
    print("=" * 60)
    print("✨ Time-based intelligent greetings")
    print("✨ FAISS-optimized vector semantic search")
    print("✨ Smart intent detection & context awareness")
    print("✨ Multi-strategy FAQ matching")
    print("✨ Enhanced conversational memory")
    print("✨ Medical safety warnings & doctor consultations")
    print("=" * 60)
    
    session_id = str(uuid.uuid4())[:8]
    print(f"Session ID: {session_id}")
    print(f"\n{TimeBasedGreeting.get_welcome_message()}")
    print("I'll understand your intent, remember our conversation, and provide reliable answers.")
    print("Type 'bye' to exit")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print(f"{BOT_NAME}: Goodbye! Take care of your health! 💖 Remember to consult a doctor for any medical concerns.")
                break
                
            start_time = time.time()
            response = bot.process_query(user_input, session_id)
            end_time = time.time()
            
            print(f"{BOT_NAME}: {response}")
            print(f"[Processed in {end_time-start_time:.2f}s]")
            print("-" * 60)
            
        except KeyboardInterrupt:
            print(f"\n{BOT_NAME}: Session ended. Stay healthy! 💪")
            break
        except Exception as e:
            print(f"{BOT_NAME}: I encountered a small issue. Please try rephrasing your question.")

if __name__ == '__main__':
    run_cli()
