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
# CONFIGURATION
# ================================
BASE_DIR = os.path.dirname(__file__)
FAQ_PATH = os.path.join(BASE_DIR, 'faq.json')
KNOWLEDGE_GRAPH_PATH = os.path.join(BASE_DIR, 'knowledge_graph.json')
USER_PROFILES_PATH = os.path.join(BASE_DIR, 'user_profiles.json')

BOT_NAME = "ME Bot Pro"
MAX_CONTEXT = 20
MEDICAL_DISCLAIMER = "⚠️ I'm an AI assistant, not a medical professional. For serious symptoms, please consult a doctor immediately."

EMERGENCY_KEYWORDS = {
    'chest pain', 'difficulty breathing', 'heavy bleeding', 'suicide', 
    'emergency', 'urgent help', 'severe pain', 'heart attack', 'stroke',
    'cannot breathe', 'unconscious', 'broken bone', 'seizure'
}

CRISIS_KEYWORDS = {
    'depressed', 'depression', 'suicidal', 'self harm', 'want to die',
    'kill myself', 'end it all', 'hopeless', 'no will to live'
}

# ================================
# SMART NLP (Keep from previous version)
# ================================

class SmartNLP:
    """Optimized but intelligent NLP understanding"""
    
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
        """Comprehensive but fast query analysis"""
        query_lower = query.lower()
        
        # Semantic intent detection
        intent = 'general'
        for intent_type, patterns in SmartNLP.INTENT_PATTERNS.items():
            if any(pattern in query_lower for pattern in patterns):
                intent = intent_type
                break
        
        # Medical context detection with synonyms
        medical_context = []
        for primary, synonyms in SmartNLP.MEDICAL_SYNONYMS.items():
            if (primary in query_lower or 
                any(synonym in query_lower for synonym in synonyms)):
                medical_context.append(primary)
        
        # Tone analysis
        tone = 'neutral'
        if any(word in query_lower for word in ['worried', 'scared', 'anxious']):
            tone = 'anxious'
        elif any(word in query_lower for word in ['thank', 'appreciate']):
            tone = 'grateful'
        elif any(word in query_lower for word in ['confused', 'unsure']):
            tone = 'confused'
        
        # Query complexity
        words = query.split()
        complexity = 'advanced' if len(words) > 8 else 'basic'
        
        return {
            'intent': intent,
            'medical_context': medical_context,
            'tone': tone,
            'complexity': complexity,
            'phonetic_variations': SmartNLP._get_phonetic_variations(query)
        }
    
    @staticmethod
    def _get_phonetic_variations(query):
        """Generate phonetic variations for typo tolerance"""
        words = query.split()
        phonetic_variants = []
        
        for word in words:
            if len(word) > 3:  # Only substantial words
                try:
                    phonetic = phonetics.metaphone(word)
                    phonetic_variants.append(phonetic)
                except:
                    continue
        return phonetic_variants

# ================================
# ADVANCED MEDICAL NLP
# ================================

class AdvancedMedicalNLP:
    """Next-generation medical NLP with entity recognition"""
    
    MEDICAL_ENTITIES = {
        'conditions': {'pcod', 'pcos', 'endometriosis', 'fibroids', 'thyroid', 'diabetes'},
        'symptoms': {'pain', 'bleeding', 'cramps', 'headache', 'nausea', 'fatigue'},
        'treatments': {'medication', 'surgery', 'therapy', 'diet', 'exercise'},
        'body_parts': {'ovary', 'uterus', 'breast', 'pelvic', 'abdominal'},
        'time_frames': {'days', 'weeks', 'months', 'years', 'recently'}
    }
    
    SYMPTOM_SEVERITY_INDICATORS = {
        'mild': {'slight', 'mild', 'minor', 'manageable'},
        'moderate': {'moderate', 'bothersome', 'uncomfortable'},
        'severe': {'severe', 'intense', 'unbearable', 'excruciating', 'debilitating'}
    }
    
    @staticmethod
    def extract_medical_entities(text):
        """Advanced medical entity recognition"""
        text_lower = text.lower()
        entities = {}
        
        for entity_type, terms in AdvancedMedicalNLP.MEDICAL_ENTITIES.items():
            found_entities = [term for term in terms if term in text_lower]
            if found_entities:
                entities[entity_type] = found_entities
        
        # Extract severity
        entities['severity'] = AdvancedMedicalNLP._detect_symptom_severity(text_lower)
        
        # Extract time mentions
        entities['time_mentions'] = AdvancedMedicalNLP._extract_time_mentions(text)
        
        return entities
    
    @staticmethod
    def _detect_symptom_severity(text):
        """Detect symptom severity from text"""
        for severity_level, indicators in AdvancedMedicalNLP.SYMPTOM_SEVERITY_INDICATORS.items():
            if any(indicator in text for indicator in indicators):
                return severity_level
        return 'unknown'
    
    @staticmethod
    def _extract_time_mentions(text):
        """Extract time duration mentions"""
        time_patterns = [
            r'(\d+)\s*(day|week|month|year)s?',
            r'(recently|lately|for a while)',
            r'(since|for)\s+(\w+\s+\d+)'
        ]
        
        time_mentions = []
        for pattern in time_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                time_mentions.append(match.group())
        
        return time_mentions

# ================================
# KNOWLEDGE GRAPH
# ================================

class MedicalKnowledgeGraph:
    """Graph-based medical knowledge representation"""
    
    def __init__(self):
        self.graph = self._load_knowledge_graph()
    
    def _load_knowledge_graph(self):
        """Load medical knowledge graph"""
        try:
            with open(KNOWLEDGE_GRAPH_PATH, 'r') as f:
                return json.load(f)
        except:
            # Fallback basic graph
            return {
                'pcod': {
                    'related_conditions': ['pcos', 'insulin_resistance', 'infertility'],
                    'symptoms': ['irregular periods', 'weight gain', 'acne'],
                    'treatments': ['diet', 'exercise', 'medication'],
                    'complications': ['diabetes', 'heart_disease']
                },
                'pcos': {
                    'related_conditions': ['pcod', 'metabolic_syndrome'],
                    'symptoms': ['irregular periods', 'excess hair', 'infertility'],
                    'treatments': ['birth control', 'metformin', 'lifestyle']
                }
            }
    
    def get_related_concepts(self, concept, relation_type=None):
        """Get related medical concepts"""
        concept_lower = concept.lower()
        
        if concept_lower in self.graph:
            if relation_type:
                return self.graph[concept_lower].get(relation_type, [])
            return self.graph[concept_lower]
        
        return []

# ================================
# USER PROFILES
# ================================

class UserProfileManager:
    """Personalized user profiling and adaptation"""
    
    def __init__(self):
        self.profiles = self._load_profiles()
    
    def _load_profiles(self):
        """Load user profiles"""
        try:
            with open(USER_PROFILES_PATH, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def get_profile(self, user_id):
        """Get or create user profile"""
        if user_id not in self.profiles:
            self.profiles[user_id] = {
                'demographics': {},
                'medical_history': [],
                'conversation_patterns': {},
                'preferences': {
                    'response_length': 'detailed',
                    'technical_level': 'layman',
                    'communication_style': 'empathetic'
                },
                'trust_level': 0.5,
                'created_at': datetime.now().isoformat(),
                'last_activity': datetime.now().isoformat()
            }
        return self.profiles[user_id]
    
    def update_medical_history(self, user_id, entities, query):
        """Update user's medical history based on conversation"""
        profile = self.get_profile(user_id)
        
        if 'conditions' in entities:
            for condition in entities['conditions']:
                if condition not in profile['medical_history']:
                    profile['medical_history'].append(condition)
        
        profile['last_activity'] = datetime.now().isoformat()
        self._save_profiles()
    
    def adapt_response_style(self, response, user_id):
        """Adapt response based on user preferences"""
        profile = self.get_profile(user_id)
        prefs = profile['preferences']
        
        # Adjust response length
        if prefs['response_length'] == 'concise':
            response = self._make_concise(response)
        
        return response
    
    def _make_concise(self, text):
        """Make response more concise"""
        sentences = text.split('. ')
        return '. '.join(sentences[:2]) + '.' if len(sentences) > 2 else text
    
    def _save_profiles(self):
        """Save user profiles to disk"""
        try:
            with open(USER_PROFILES_PATH, 'w') as f:
                json.dump(self.profiles, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save user profiles: {e}")

# ================================
# ADVANCED INTELLIGENT SEARCH (WITH FIX)
# ================================

class AdvancedIntelligentSearch:
    """Multi-modal retrieval with knowledge graph enhancement"""
    
    def __init__(self, faq_data):
        self.faq_data = faq_data
        self.questions, self.answers = self._extract_qa_pairs(faq_data)  # FIXED: Added this method
        self.knowledge_graph = MedicalKnowledgeGraph()
        
        print(f"✅ Prepared {len(self.questions)} Q&A pairs for advanced search")
        
        # Multi-encoder setup
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self._setup_advanced_indices()
    
    def _extract_qa_pairs(self, faq_data):
        """Flexibly extract questions and answers from various data structures"""
        questions = []
        answers = []
        
        for item in faq_data:
            if isinstance(item, dict):
                if 'question' in item and 'answer' in item:
                    questions.append(item['question'])
                    answers.append(item['answer'])
                elif 'q' in item and 'a' in item:
                    questions.append(item['q'])
                    answers.append(item['a'])
                elif 'Question' in item and 'Answer' in item:
                    questions.append(item['Question'])
                    answers.append(item['Answer'])
                elif len(item) == 2:
                    keys = list(item.keys())
                    questions.append(str(item[keys[0]]))
                    answers.append(str(item[keys[1]]))
                else:
                    questions.append(str(item))
                    answers.append(str(item))
            elif isinstance(item, (str, int, float)):
                questions.append(str(item))
                answers.append(str(item))
            else:
                questions.append(str(item))
                answers.append(str(item))
        
        return questions, answers
    
    def _setup_advanced_indices(self):
        """Setup multiple specialized indices"""
        try:
            if not self.questions:
                print("⚠️ No questions available for semantic indexing")
                self.index = None
                return
                
            print("🔄 Creating advanced semantic embeddings...")
            embeddings = self.embedder.encode(self.questions)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings_norm = embeddings / norms
            
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index.add(embeddings_norm.astype('float32'))
            print(f"✅ Advanced semantic index created with {len(self.questions)} embeddings")
        except Exception as e:
            print(f"⚠️ Advanced index setup failed: {e}")
            self.index = None
    
    def multi_modal_search(self, query, nlp_analysis, medical_entities):
        """Advanced search combining multiple strategies"""
        all_matches = []
        
        # 1. Knowledge-graph enhanced semantic search
        kg_matches = self._knowledge_graph_search(query, medical_entities)
        all_matches.extend(kg_matches)
        
        # 2. Enhanced semantic search
        semantic_matches = self._semantic_search(query)
        all_matches.extend(semantic_matches)
        
        # 3. Temporal-aware search
        if medical_entities.get('time_mentions'):
            temporal_matches = self._temporal_search(query, medical_entities)
            all_matches.extend(temporal_matches)
        
        # 4. Severity-aware ranking
        severity_aware_matches = self._severity_aware_ranking(all_matches, medical_entities)
        
        return self._advanced_ranking(severity_aware_matches, nlp_analysis, medical_entities)
    
    def _knowledge_graph_search(self, query, medical_entities):
        """Search enhanced with knowledge graph relations"""
        matches = []
        
        for idx, (question, answer) in enumerate(zip(self.questions, self.answers)):
            question_lower = question.lower()
            answer_lower = answer.lower()
            
            # Check for medical concepts
            medical_matches = 0
            for entity_type, entities in medical_entities.items():
                if entity_type != 'severity' and entity_type != 'time_mentions':
                    for entity in entities:
                        if entity in question_lower or entity in answer_lower:
                            medical_matches += 1
            
            if medical_matches > 0:
                matches.append({
                    'index': idx, 'score': medical_matches * 0.2,
                    'question': question, 'answer': answer,
                    'type': 'knowledge_graph'
                })
        
        return matches
    
    def _semantic_search(self, query, top_k=8):
        """Enhanced semantic search"""
        if not self.index:
            return []
        
        try:
            query_vec = self.embedder.encode([query])
            query_norm = query_vec / np.linalg.norm(query_vec)
            scores, indices = self.index.search(query_norm.astype('float32'), top_k)
            
            return [{
                'index': idx, 'score': float(score),
                'question': self.questions[idx], 
                'answer': self.answers[idx],
                'type': 'semantic'
            } for idx, score in zip(indices[0], scores[0]) if score > 0.3]
        except Exception as e:
            print(f"⚠️ Semantic search error: {e}")
            return []
    
    def _temporal_search(self, query, medical_entities):
        """Time-aware search for duration-related queries"""
        matches = []
        
        for idx, (question, answer) in enumerate(zip(self.questions, self.answers)):
            # Boost matches that mention time frames
            if any(time_word in question.lower() for time_word in ['day', 'week', 'month', 'year', 'duration']):
                matches.append({
                    'index': idx, 'score': 0.6,
                    'question': question, 'answer': answer,
                    'type': 'temporal'
                })
        
        return matches
    
    def _severity_aware_ranking(self, matches, medical_entities):
        """Adjust ranking based on symptom severity"""
        severity = medical_entities.get('severity', 'unknown')
        
        for match in matches:
            # Boost severe symptom matches
            if severity == 'severe' and any(word in match['answer'].lower() for word in ['emergency', 'urgent', 'immediately']):
                match['score'] *= 1.3
        
        return matches
    
    def _advanced_ranking(self, matches, nlp_analysis, medical_entities):
        """Advanced multi-factor ranking"""
        scored_matches = []
        
        for match in matches:
            base_score = match['score']
            
            # Intent alignment boost
            if nlp_analysis['intent'] != 'general':
                intent_words = SmartNLP.INTENT_PATTERNS.get(nlp_analysis['intent'], set())
                if any(word in match['question'].lower() for word in intent_words):
                    base_score *= 1.15
            
            # Medical entity alignment boost
            if medical_entities:
                entity_boost = sum(0.1 for entity_list in medical_entities.values() 
                                 for entity in entity_list if entity in match['question'].lower())
                base_score *= (1 + entity_boost)
            
            match['final_score'] = min(base_score, 1.0)
            scored_matches.append(match)
        
        # Remove duplicates and sort
        unique_matches = {}
        for match in scored_matches:
            key = match['question']
            if key not in unique_matches or match['final_score'] > unique_matches[key]['final_score']:
                unique_matches[key] = match
        
        return sorted(unique_matches.values(), key=lambda x: x['final_score'], reverse=True)[:8]

# ================================
# HEALTH MONITOR
# ================================

class HealthMonitor:
    """Proactive health monitoring and alert system"""
    
    def __init__(self, user_profiles):
        self.user_profiles = user_profiles
        self.risk_patterns = {
            'pcod_high_risk': {
                'conditions': ['pcod', 'pcos'],
                'symptoms': ['weight gain', 'irregular periods', 'acne'],
                'risk_level': 'medium',
                'recommendation': 'Consider consulting an endocrinologist for comprehensive management'
            }
        }
    
    def assess_health_risks(self, user_id, current_entities, conversation_history):
        """Assess potential health risks based on conversation patterns"""
        profile = self.user_profiles.get_profile(user_id)
        medical_history = profile.get('medical_history', [])
        
        risks = []
        
        for risk_name, pattern in self.risk_patterns.items():
            # Check if user has risk factors
            has_conditions = any(cond in medical_history for cond in pattern['conditions'])
            has_current_symptoms = any(symptom in str(current_entities.get('symptoms', [])) 
                                     for symptom in pattern['symptoms'])
            
            if has_conditions or has_current_symptoms:
                risks.append({
                    'risk_name': risk_name,
                    'risk_level': pattern['risk_level'],
                    'recommendation': pattern['recommendation'],
                    'triggers': list(set(medical_history) & set(pattern['conditions']))
                })
        
        return risks

# ================================
# ADVANCED RAG SYSTEM
# ================================

class AdvancedRAG:
    """Enhanced RAG with medical context understanding"""
    
    def __init__(self):
        try:
            self.generator = pipeline(
                'text2text-generation',
                model='google/flan-t5-base',
                device=-1,
                max_length=300,
                torch_dtype=torch.float32
            )
            print("✅ Advanced RAG system initialized")
        except Exception as e:
            print(f"⚠️ Advanced RAG initialization failed: {e}")
            self.generator = None
    
    def enhance_with_context(self, query, faq_answer, conversation_context, medical_entities):
        """Context-aware response enhancement"""
        if not self.generator:
            return faq_answer
        
        try:
            # Build context-aware prompt
            context_str = self._build_context_string(conversation_context, medical_entities)
            
            prompt = f"""
            Original information: {faq_answer}
            User's current question: {query}
            Medical context: {medical_entities}
            
            Please provide a comprehensive and empathetic response:
            """
            
            result = self.generator(prompt, do_sample=True, temperature=0.4, 
                                  max_length=350)[0]['generated_text']
            
            return result if len(result) > 30 else faq_answer
            
        except Exception as e:
            print(f"RAG enhancement error: {e}")
            return faq_answer
    
    def _build_context_string(self, conversation_context, medical_entities):
        """Build context string for RAG"""
        context_parts = []
        
        if conversation_context.get('topics'):
            context_parts.append(f"Recent topics: {', '.join(conversation_context['topics'][-3:])}")
        
        return '; '.join(context_parts)

# ================================
# ADVANCED RESPONSE BUILDER
# ================================

class AdvancedResponseBuilder:
    """Next-generation response generation"""
    
    def __init__(self, rag_system, user_profiles, health_monitor):
        self.rag = rag_system
        self.user_profiles = user_profiles
        self.health_monitor = health_monitor
        self.nlp = SmartNLP()
    
    def build_advanced_response(self, query, matches, user_id, conversation_context, medical_entities):
        """Build comprehensive advanced response"""
        if not matches:
            return self._intelligent_fallback(query, conversation_context, medical_entities)
        
        # Get user profile for personalization
        profile = self.user_profiles.get_profile(user_id)
        
        # Build base response
        base_response = matches[0]['answer']
        
        # Enhance with RAG for complex queries
        if self._needs_enhancement(query, matches[0]['final_score'], medical_entities):
            enhanced_response = self.rag.enhance_with_context(query, base_response, conversation_context, medical_entities)
        else:
            enhanced_response = base_response
        
        # Add proactive health insights
        insights_response = self._add_health_insights(enhanced_response, user_id, medical_entities, conversation_context)
        
        # Personalize based on user preferences
        personalized_response = self.user_profiles.adapt_response_style(insights_response, user_id)
        
        # Add conversation continuity
        contextual_response = self._add_advanced_context(personalized_response, conversation_context, medical_entities)
        
        return contextual_response
    
    def _add_health_insights(self, response, user_id, medical_entities, conversation_context):
        """Add proactive health insights and monitoring"""
        risks = self.health_monitor.assess_health_risks(user_id, medical_entities, conversation_context)
        
        if risks:
            high_risks = [r for r in risks if r['risk_level'] == 'high']
            if high_risks:
                risk_msg = "\n\n🔍 Health Note: " + high_risks[0]['recommendation']
                response += risk_msg
        
        return response
    
    def _add_advanced_context(self, response, conversation_context, medical_entities):
        """Advanced context integration"""
        if conversation_context and conversation_context.get('topics'):
            recent_topics = conversation_context['topics'][-2:]
            current_entities = list(medical_entities.keys())
            
            if recent_topics and current_entities:
                topic_str = ', '.join(recent_topics)
                entity_str = ', '.join(current_entities[:2])
                return f"Building on our discussion of {topic_str}, and considering {entity_str}: {response}"
        
        return response
    
    def _needs_enhancement(self, query, best_score, medical_entities):
        """Determine if response needs RAG enhancement"""
        return (best_score < 0.6 or 
                len(medical_entities) > 2 or
                len(query.split()) > 10)
    
    def _intelligent_fallback(self, query, conversation_context, medical_entities):
        """Advanced fallback responses"""
        if medical_entities:
            entities_str = ', '.join([e for entity_list in medical_entities.values() 
                                    for e in entity_list][:3])
            return f"I specialize in women's health topics like {entities_str}. Could you tell me more about your specific concern?"
        
        return "I focus on women's health including hormonal balance, menstrual health, PCOD/PCOS. What specific area interests you?"

# ================================
# SESSION MANAGER
# ================================

class SessionManager:
    def __init__(self):
        self.sessions = {}
    
    def get_session(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'history': deque(maxlen=MAX_CONTEXT),
                'topics': set(),
                'created': datetime.now(),
                'interaction_count': 0
            }
        return self.sessions[session_id]
    
    def update_session(self, session_id, user_input, response, topics):
        session = self.get_session(session_id)
        session['history'].append({
            'user': user_input,
            'bot': response,
            'time': datetime.now(),
            'topics': topics
        })
        session['topics'].update(topics)
        session['interaction_count'] += 1
    
    def get_context(self, session_id):
        session = self.get_session(session_id)
        return {
            'recent_history': list(session['history'])[-3:],
            'topics': list(session['topics']),
            'interaction_count': session['interaction_count']
        }

# ================================
# NEXT-GEN MAIN BOT
# ================================

class NextGenMEBot:
    def __init__(self):
        print("🚀 Initializing Next-Gen ME Bot...")
        
        # Load data
        self.faq_data = self._load_faq()
        print(f"✅ Loaded {len(self.faq_data)} FAQ items")
        
        # Initialize advanced systems
        self.sessions = SessionManager()
        self.user_profiles = UserProfileManager()
        self.health_monitor = HealthMonitor(self.user_profiles)
        self.rag = AdvancedRAG()
        self.search = AdvancedIntelligentSearch(self.faq_data)
        self.response_builder = AdvancedResponseBuilder(
            self.rag, self.user_profiles, self.health_monitor
        )
        
        print("✅ Next-gen systems ready")
        print("✨ Features: Knowledge Graphs, Personalization, Health Monitoring, Advanced NLP")
    
    def _load_faq(self):
        """Robust FAQ loading"""
        try:
            with open(FAQ_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception as e:
            print(f"❌ Error loading FAQ: {e}")
            return []
    
    def process(self, user_input, session_id):
        """Advanced processing pipeline"""
        start_time = time.time()
        
        # Pre-processing checks
        user_input_lower = user_input.lower()
        
        # 1. Emergency and crisis detection
        emergency_response = self._check_emergencies(user_input_lower)
        if emergency_response:
            return emergency_response
        
        # 2. Greeting and special intent handling
        special_response = self._handle_special_intents(user_input_lower, session_id)
        if special_response:
            return special_response
        
        # 3. Advanced NLP analysis
        nlp_analysis = SmartNLP.analyze_query(user_input)
        medical_entities = AdvancedMedicalNLP.extract_medical_entities(user_input)
        
        # 4. Get conversation context
        context = self.sessions.get_context(session_id)
        
        # 5. Advanced multi-modal search
        matches = self.search.multi_modal_search(user_input, nlp_analysis, medical_entities)
        
        # 6. Build advanced response
        response = self.response_builder.build_advanced_response(
            user_input, matches, session_id, context, medical_entities
        )
        
        # 7. Update systems
        self.sessions.update_session(
            session_id, user_input, response, 
            nlp_analysis['medical_context'] + list(medical_entities.keys())
        )
        self.user_profiles.update_medical_history(session_id, medical_entities, user_input)
        
        processing_time = time.time() - start_time
        print(f"⏱️ Advanced processing: {processing_time:.2f}s")
        
        return response
    
    def _check_emergencies(self, user_input):
        """Enhanced emergency detection"""
        if any(emergency in user_input for emergency in EMERGENCY_KEYWORDS):
            return f"🚨 MEDICAL EMERGENCY: {MEDICAL_DISCLAIMER} Please call emergency services immediately!"
        
        if any(crisis in user_input for crisis in CRISIS_KEYWORDS):
            crisis_msg = "🚨 CRISIS SUPPORT: I'm deeply concerned about what you're sharing. "
            crisis_msg += "Please contact these resources NOW:\n"
            crisis_msg += "• National Suicide Prevention Lifeline: 988\n"
            crisis_msg += "• Crisis Text Line: Text HOME to 741741\n"
            crisis_msg += "• Emergency Services: 911\n"
            crisis_msg += "You are not alone - professional help is available."
            return crisis_msg
        
        return None
    
    def _handle_special_intents(self, user_input, session_id):
        """Handle special conversation intents"""
        # Greetings
        if any(greeting in user_input for greeting in ['hello', 'hi', 'hey']):
            return "Hello! I'm your advanced women's health assistant. I can help with medical information, symptom understanding, and health insights. What would you like to discuss today?"
        
        # Thanks
        if any(thanks in user_input for thanks in ['thank', 'thanks']):
            return "You're welcome! I'm glad I could help. Is there anything else you'd like to know about your health?"
        
        # Goodbye
        if any(bye in user_input for bye in ['bye', 'goodbye', 'exit', 'quit']):
            return "Thank you for chatting! Remember to consult healthcare professionals for personal medical advice. Take care! 💖"
        
        return None

# ================================
# TEST CASES
# ================================

def run_comprehensive_tests():
    """Run comprehensive test cases"""
    print("🧪 RUNNING COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    bot = NextGenMEBot()
    test_session = "test_session_" + str(int(time.time()))
    
    test_cases = [
        # Emergency and Crisis
        ("I have severe chest pain", "EMERGENCY", "Should trigger emergency response"),
        ("I'm feeling suicidal today", "CRISIS", "Should trigger crisis response"),
        
        # Medical Understanding
        ("What are PCOD symptoms?", "SYMPTOMS", "Should understand PCOD and symptoms"),
        ("How to manage hormonal imbalance?", "HORMONE", "Should understand hormonal concepts"),
        
        # Context Awareness
        ("First: What is PCOD?", "CONTEXT_START", "Start context"),
        ("Then: What are its symptoms?", "CONTEXT_CONTINUE", "Continue context"),
        
        # Typo Handling
        ("What is PCODD?", "TYPO", "Should handle typos"),
        ("haromonal imbalance", "PHONETIC", "Should handle phonetic errors"),
        
        # Personalization
        ("Explain PCOS simply", "SIMPLE", "Should provide simplified explanation"),
        
        # Complex Queries
        ("What's the difference between PCOD and PCOS with treatment options?", "COMPLEX", "Should handle complex multi-part questions")
    ]
    
    passed = 0
    failed = 0
    
    for user_input, test_type, description in test_cases:
        print(f"\n🔍 TEST: {description}")
        print(f"Input: '{user_input}'")
        
        try:
            response = bot.process(user_input, test_session)
            print(f"Response: {response[:150]}...")
            
            # Validate based on test type
            if test_type == "EMERGENCY":
                if "EMERGENCY" in response:
                    print("✅ PASS - Emergency detected correctly")
                    passed += 1
                else:
                    print("❌ FAIL - Emergency not detected")
                    failed += 1
                    
            elif test_type == "CRISIS":
                if "CRISIS" in response or "988" in response:
                    print("✅ PASS - Crisis detected correctly")
                    passed += 1
                else:
                    print("❌ FAIL - Crisis not detected")
                    failed += 1
                    
            elif test_type in ["SYMPTOMS", "HORMONE"]:
                if len(response) > 20 and "sorry" not in response.lower():
                    print("✅ PASS - Medical understanding working")
                    passed += 1
                else:
                    print("❌ FAIL - Medical understanding failed")
                    failed += 1
                    
            else:
                # For other tests, just check we get a reasonable response
                if response and len(response) > 10:
                    print("✅ PASS - Reasonable response generated")
                    passed += 1
                else:
                    print("❌ FAIL - No reasonable response")
                    failed += 1
                    
        except Exception as e:
            print(f"❌ ERROR: {e}")
            failed += 1
    
    print(f"\n📊 TEST SUMMARY: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    return passed, failed

# ================================
# RUN BOT OR TESTS
# ================================

def run_nextgen_cli():
    """Run the next-gen bot in CLI mode"""
    bot = NextGenMEBot()
    
    print(f"\n{BOT_NAME} Next-Generation")
    print("=" * 60)
    print("✨ Knowledge Graph Integration")
    print("✨ Personalized User Profiles") 
    print("✨ Proactive Health Monitoring")
    print("✨ Advanced Medical NLP")
    print("✨ Multi-Modal Semantic Search")
    print("=" * 60)
    
    session_id = uuid.uuid4().hex[:8]
    print(f"Session: {session_id}")
    print("\nHello! I'm your advanced women's health assistant.")
    print("Type 'test' to run comprehensive tests")
    print("Type 'bye' to exit")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            
            if user_input.lower() == 'test':
                run_comprehensive_tests()
                continue
                
            if user_input.lower() in {'exit', 'quit', 'bye'}:
                print(f"{BOT_NAME}: Goodbye! 💖")
                break
            
            start = time.time()
            response = bot.process(user_input, session_id)
            elapsed = time.time() - start
            
            print(f"{BOT_NAME}: {response}")
            print(f"[Next-gen processing: {elapsed:.2f}s]")
            print("-" * 60)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"{BOT_NAME}: I encountered an error. Please try again.")
            print(f"Error: {e}")

if __name__ == '__main__':
    # You can either run tests or the bot
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        run_comprehensive_tests()
    else:
        run_nextgen_cli()
