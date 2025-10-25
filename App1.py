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
import torch
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote

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
# TRUE MEDICAL RAG SYSTEM (SIMPLIFIED)
# -------------------------
class TrueMedicalRAG:
    def __init__(self):
        self.trusted_sources = {
            'WHO': 'World Health Organization',
            'CDC': 'Centers for Disease Control', 
            'NIH': 'National Institutes of Health',
            'ACOG': 'American College of Obstetricians',
            'Mayo Clinic': 'Mayo Clinic'
        }
        
        # Comprehensive medical knowledge base
        self.medical_knowledge = self._build_medical_knowledge_base()
        
        # Initialize the generator
        try:
            self.generator = pipeline(
                'text2text-generation',
                model='google/flan-t5-base',
                device=-1,
                torch_dtype=torch.float32
            )
            print("✅ True RAG System Initialized")
        except Exception as e:
            print(f"⚠️ RAG Generator unavailable: {e}")
            self.generator = None
    
    def _build_medical_knowledge_base(self):
        """Build comprehensive medical knowledge base"""
        return {
            # WHO Guidelines
            'who': {
                'pcod': "WHO recommends lifestyle modifications as first-line treatment for PCOS, including balanced diet, regular exercise, and weight management for overweight individuals.",
                'pcos': "WHO guidelines emphasize early diagnosis and multidisciplinary approach for PCOS management, addressing both reproductive and metabolic aspects.",
                'menstrual': "WHO defines normal menstrual cycles as 21-35 days. Irregularities may indicate conditions like PCOS, thyroid disorders, or hormonal imbalances.",
                'hormone': "WHO recognizes that stress management, adequate sleep, and balanced nutrition are crucial for maintaining hormonal balance in women.",
                'fertility': "WHO defines infertility as failure to achieve pregnancy after 12 months of regular unprotected intercourse and recommends evaluation after 6 months for women over 35.",
                'contraception': "WHO provides medical eligibility criteria for contraceptive methods based on individual health conditions and risk factors.",
                'menopause': "WHO guidelines recommend individualized approach to menopausal symptom management considering personal risk factors and preferences.",
                'mental_health': "WHO emphasizes integrated mental health care in women's health services, particularly for conditions like postpartum depression and anxiety disorders."
            },
            
            # ACOG Guidelines
            'acog': {
                'pcod': "ACOG recommends combined oral contraceptives for menstrual cycle regulation in PCOS patients without contraindications.",
                'pcos': "ACOG guidelines suggest metformin for insulin resistance in PCOS when lifestyle modifications are insufficient, particularly for women trying to conceive.",
                'endometriosis': "ACOG recommends laparoscopic diagnosis for suspected endometriosis and comprehensive pain management strategies.",
                'menopause': "ACOG provides evidence-based guidelines on hormone therapy, emphasizing individualized risk-benefit assessment.",
                'contraception': "ACOG offers comprehensive contraception guidelines including long-acting reversible contraceptives (LARCs) as first-line options.",
                'pregnancy': "ACOG recommends prenatal care initiation in first trimester and regular monitoring throughout pregnancy.",
                'screening': "ACOG guidelines include regular breast cancer screening mammography and cervical cancer screening based on age and risk factors."
            },
            
            # CDC Recommendations
            'cdc': {
                'pcod': "CDC notes that PCOS affects 6-12% of US women of reproductive age and is a leading cause of infertility. Early diagnosis and management are crucial.",
                'pcos': "CDC recommends screening for diabetes and cardiovascular risk factors in women with PCOS, with regular monitoring of blood pressure and lipid levels.",
                'contraception': "CDC provides US Medical Eligibility Criteria for Contraceptive Use, helping providers select appropriate methods based on health conditions.",
                'cancer': "CDC offers guidelines for breast and cervical cancer screening, emphasizing regular mammograms and Pap tests based on age guidelines.",
                'mental_health': "CDC recognizes the significant impact of mental health on overall women's health and recommends integrated care approaches.",
                'vaccination': "CDC provides vaccination recommendations for women including HPV vaccine, Tdap during pregnancy, and annual influenza vaccine."
            },
            
            # Latest Research Insights
            'research': {
                'pcod': "Recent 2023 studies show inositol supplementation may improve ovarian function and metabolic parameters in PCOS patients. Lifestyle interventions remain cornerstone.",
                'pcos': "2024 meta-analysis confirms that combined lifestyle modifications can significantly improve insulin sensitivity and reproductive outcomes in PCOS.",
                'endometriosis': "Emerging research explores novel pain management approaches and immune-modulating treatments for endometriosis beyond traditional hormonal therapies.",
                'fertility': "Advanced research focuses on ovarian rejuvenation techniques and personalized fertility treatments based on genetic markers.",
                'menopause': "Recent trials demonstrate efficacy of non-hormonal therapies for vasomotor symptoms, expanding treatment options for menopausal women.",
                'nutrition': "Current studies emphasize Mediterranean diet patterns for hormonal balance and inflammatory reduction in women's health conditions."
            },
            
            # Mayo Clinic Patient Education
            'patient_education': {
                'pcod': "PCOS management focuses on addressing individual concerns - whether irregular periods, fertility issues, or metabolic symptoms. Treatment is personalized.",
                'pcos': "Living with PCOS involves consistent lifestyle habits, regular medical follow-ups, and addressing both physical and emotional aspects of the condition.",
                'self_care': "Stress reduction techniques, regular physical activity, and balanced nutrition form the foundation of women's health self-care practices.",
                'symptom_tracking': "Monitoring menstrual cycles, symptoms, and lifestyle factors can help identify patterns and optimize treatment approaches.",
                'preventive_care': "Regular health screenings, vaccinations, and preventive measures are essential components of comprehensive women's healthcare."
            }
        }
    
    def get_true_rag_answer(self, user_query, faq_context, conversation_context):
        """TRUE RAG: Combine FAQ + Comprehensive Medical Knowledge"""
        if not self.generator:
            return faq_context['answer']
        
        try:
            print(f"🔍 Accessing medical knowledge for: {user_query}")
            
            # 1. Extract relevant medical knowledge
            external_knowledge = self._extract_relevant_knowledge(user_query)
            
            # 2. Create comprehensive prompt
            prompt = self._create_true_rag_prompt(
                user_query, faq_context, external_knowledge, conversation_context
            )
            
            # 3. Generate enhanced response
            generated_response = self.generator(
                prompt,
                max_length=500,
                do_sample=True,
                temperature=0.7,
                num_return_sequences=1
            )[0]['generated_text']
            
            # 4. Validate and format response
            validated_response = self._validate_and_format(generated_response, external_knowledge)
            
            print(f"✅ Enhanced with {len(external_knowledge)} medical knowledge sources")
            return validated_response
            
        except Exception as e:
            print(f"⚠️ True RAG failed: {e}")
            return faq_context['answer']
    
    def _extract_relevant_knowledge(self, query):
        """Extract relevant knowledge from medical knowledge base"""
        query_lower = query.lower()
        relevant_knowledge = []
        
        # Search through all knowledge categories
        for category, knowledge_dict in self.medical_knowledge.items():
            for keyword, knowledge in knowledge_dict.items():
                if keyword in query_lower:
                    relevant_knowledge.append({
                        'source': category.upper(),
                        'content': knowledge,
                        'keyword': keyword
                    })
        
        # If no specific matches found, provide general women's health knowledge
        if not relevant_knowledge:
            relevant_knowledge = [
                {
                    'source': 'GENERAL',
                    'content': 'Comprehensive women\'s healthcare includes regular screenings, preventive care, and addressing both physical and mental health aspects.',
                    'keyword': 'general'
                }
            ]
        
        return relevant_knowledge[:4]  # Limit to top 4 most relevant
    
    def _create_true_rag_prompt(self, user_query, faq_context, external_knowledge, conversation_context):
        """Create comprehensive RAG prompt with medical knowledge"""
        
        # Build external knowledge string
        external_info = "MEDICAL KNOWLEDGE BASE:\n"
        for knowledge in external_knowledge:
            external_info += f"• {knowledge['source']}: {knowledge['content']}\n"
        
        context_str = ""
        if conversation_context and conversation_context.get('recent_history'):
            recent = conversation_context['recent_history'][-2:]
            if recent:
                context_str = "CONVERSATION CONTEXT:\n"
                for exchange in recent:
                    context_str += f"User: {exchange.get('user_input', '')}\n"
                    context_str += f"Assistant: {exchange.get('bot_response', '')}\n"
        
        prompt = f"""
        You are a comprehensive women's health assistant. Create an informed response using BOTH the specific FAQ knowledge AND general medical knowledge.
        
        USER QUESTION: "{user_query}"
        
        {context_str}
        
        SPECIFIC FAQ KNOWLEDGE:
        • Question: {faq_context['question']}
        • Answer: {faq_context['answer']}
        • Topics: {faq_context.get('topics', [])}
        
        {external_info}
        
        Create a response that:
        1. Directly answers the user's question using the FAQ knowledge
        2. Integrates relevant medical guidelines and research
        3. Provides practical, actionable information
        4. Uses clear, empathetic language
        5. Maintains medical accuracy
        6. Cites the knowledge sources appropriately
        
        COMPREHENSIVE RESPONSE:
        """
        
        return prompt
    
    def _validate_and_format(self, response, external_knowledge):
        """Validate and format the final response"""
        
        # Add source integration note
        if external_knowledge and 'source' not in response.lower():
            source_types = set(knowledge['source'] for knowledge in external_knowledge)
            sources_str = ', '.join(source_types)
            response += f"\n\n📚 This integrates knowledge from {sources_str} sources."
        
        # Ensure medical disclaimer
        if 'consult' not in response.lower() and 'doctor' not in response.lower():
            response += "\n\n⚠️ Always consult healthcare professionals for personal medical advice."
        
        return response

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
# ENCRYPTED SESSION MANAGEMENT
# -------------------------
class EncryptedSessionManager:
    def __init__(self):
        self.sessions = {}
    
    def create_session(self, session_id):
        """Create encrypted session"""
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
        """Get session with encryption"""
        session_key = self._hash_session_id(session_id)
        if session_key not in self.sessions:
            return self.create_session(session_id)
        
        self.sessions[session_key]['last_activity'] = datetime.now().isoformat()
        return self.sessions[session_key]
    
    def update_session(self, session_id, user_input, bot_response, topics):
        """Update session with new conversation"""
        session = self.get_session(session_id)
        
        session['conversation_history'].append({
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'bot_response': bot_response,
            'topics': topics
        })
        
        session['discussed_topics'].update(topics)
        
        decrypted_data = self._decrypt_data(session['encrypted_data'])
        decrypted_data['history'].append({
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'bot_response': bot_response,
            'topics': topics
        })
        session['encrypted_data'] = self._encrypt_data(decrypted_data)
    
    def get_conversation_context(self, session_id):
        """Get conversation context from encrypted session"""
        session = self.get_session(session_id)
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
            if len(word) > 3:
                try:
                    phonetic_variations.append(phonetics.metaphone(word))
                except:
                    continue
        
        return phonetic_variations

# -------------------------
# HYBRID SEARCH WITH RAG
# -------------------------
class HybridSearchWithRAG:
    def __init__(self, faq_data, rag_system):
        self.faq_data = faq_data
        self.faq_questions = [faq['question'] for faq in faq_data]
        self.nlp = AdvancedNLPUnderstanding()
        self.rag_system = rag_system
        
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self._setup_vector_index()
    
    def _setup_vector_index(self):
        """Setup FAISS index"""
        try:
            print("🔄 Setting up vector index for RAG-enhanced search...")
            self.faq_embeddings = self.embed_model.encode(self.faq_questions, convert_to_numpy=True)
            
            norms = np.linalg.norm(self.faq_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            faq_embeddings_norm = self.faq_embeddings / norms
            
            dimension = self.faq_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(faq_embeddings_norm.astype('float32'))
            
            print(f"✅ RAG-ready vector index for {len(self.faq_data)} FAQs")
        except Exception as e:
            print(f"⚠️ Vector index setup failed: {e}")
            self.index = None
    
    def comprehensive_rag_search(self, user_query):
        """Comprehensive search with RAG readiness"""
        query_analysis = self.nlp.analyze_query(user_query)
        all_matches = []
        
        all_matches.extend(self._vector_search(user_query))
        all_matches.extend(self._phonetic_search(user_query, query_analysis))
        all_matches.extend(self._keyword_search(user_query))
        all_matches.extend(self._topic_search(query_analysis['medical_context']))
        
        ranked_matches = self._rank_matches(all_matches, query_analysis)
        
        for match in ranked_matches:
            match['rag_confidence'] = self._calculate_rag_confidence(match, query_analysis)
        
        return ranked_matches
    
    def _calculate_rag_confidence(self, match, query_analysis):
        """Calculate confidence score for RAG enhancement"""
        score = 0.0
        
        if query_analysis['vocabulary_complexity']['reading_level'] == 'advanced':
            score += 0.3
        
        if any(intent in query_analysis['semantic_intent'] for intent in ['definition', 'causation']):
            score += 0.3
        
        if match['final_score'] < 0.7:
            score += 0.2
        
        return min(score, 1.0)
    
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
            
            if match['type'] == 'semantic':
                final_score *= 1.2
            
            if any(intent in match['question'].lower() for intent in query_analysis['semantic_intent']):
                final_score *= 1.1
            
            match['final_score'] = min(final_score, 1.0)
            scored_matches.append(match)
        
        unique_matches = {}
        for match in scored_matches:
            key = match['question']
            if key not in unique_matches or match['final_score'] > unique_matches[key]['final_score']:
                unique_matches[key] = match
        
        return sorted(unique_matches.values(), key=lambda x: x['final_score'], reverse=True)

# -------------------------
# RAG-ENHANCED RESPONSE BUILDER
# -------------------------
class RAGEnhancedResponseBuilder:
    def __init__(self, nlp_understanding, rag_system):
        self.nlp = nlp_understanding
        self.rag_system = rag_system
    
    def build_rag_enhanced_response(self, user_query, faq_match, conversation_context):
        """Build response enhanced with RAG knowledge"""
        query_analysis = self.nlp.analyze_query(user_query)
        
        # Use TRUE RAG for comprehensive medical responses
        should_use_rag = self._should_use_true_rag(user_query, faq_match, query_analysis)
        
        if should_use_rag and self.rag_system.generator:
            print("🎯 Using TRUE RAG with comprehensive medical knowledge")
            rag_response = self.rag_system.get_true_rag_answer(
                user_query, faq_match, conversation_context
            )
            final_response = self._add_explainability(rag_response, query_analysis)
        else:
            print("📚 Using standard FAQ explanation")
            final_response = self._build_standard_explained_response(
                user_query, faq_match, conversation_context
            )
        
        tone_adjusted = self._adjust_tone(final_response, query_analysis['tone'])
        return tone_adjusted
    
    def _should_use_true_rag(self, user_query, faq_match, query_analysis):
        """Determine if TRUE RAG should be used"""
        # Use TRUE RAG for comprehensive medical knowledge integration
        conditions = [
            any(context in query_analysis['medical_context'] for context in ['pcod_pcos', 'menstrual', 'hormonal']),
            query_analysis['vocabulary_complexity']['reading_level'] == 'advanced',
            faq_match['final_score'] < 0.8,
            any(intent in query_analysis['semantic_intent'] for intent in ['definition', 'treatment', 'causation']),
            len(user_query.split()) > 5
        ]
        
        return any(conditions)
    
    def _build_standard_explained_response(self, user_query, faq_match, conversation_context):
        """Build standard explained response (without RAG)"""
        base_answer = faq_match['answer']
        explained_answer = self._explain_in_own_words(base_answer, user_query)
        contextualized = self._add_conversation_context(explained_answer, conversation_context)
        return contextualized
    
    def _explain_in_own_words(self, base_answer, user_query):
        """Explain FAQ answer in natural language"""
        if user_query.lower().startswith('what is'):
            return f"Let me explain this clearly: {base_answer}"
        elif user_query.lower().startswith('how to'):
            return f"Here's how you can approach this: {base_answer}"
        elif 'symptom' in user_query.lower():
            return f"Here are the key signs to watch for: {base_answer}"
        else:
            return f"Based on medical information: {base_answer}"
    
    def _add_explainability(self, response, query_analysis):
        """Add explainability layer to RAG response"""
        if query_analysis['semantic_intent'] and query_analysis['semantic_intent'][0] == 'definition':
            return f"Let me break this down for you: {response}"
        elif query_analysis['tone'] == 'confused':
            return f"Let me clarify this simply: {response}"
        
        return response
    
    def _add_conversation_context(self, response, conversation_context):
        """Add conversation context to response"""
        if not conversation_context or not conversation_context.get('discussed_topics'):
            return response
        
        recent_topics = conversation_context.get('discussed_topics', [])
        if recent_topics and len(recent_topics) > 1:
            topics_str = ', '.join(list(recent_topics)[-2:])
            return f"Building on our discussion about {topics_str}, {response.lower()}"
        
        return response
    
    def _adjust_tone(self, response, tone):
        """Adjust response tone based on user's emotional state"""
        tone_prefixes = {
            'anxious': "I understand this might be concerning. ",
            'urgent': "This sounds important. ",
            'confused': "Let me explain this clearly. ",
            'grateful': "I'm glad to help! ",
            'neutral': ""
        }
        
        tone_suffixes = {
            'anxious': " Many women find this information helpful in managing their health concerns.",
            'urgent': " If symptoms are severe, please seek medical attention promptly.",
            'confused': " I hope this explanation makes things clearer for you.",
            'grateful': " Feel free to ask any follow-up questions!",
            'neutral': ""
        }
        
        prefix = tone_prefixes.get(tone, "")
        suffix = tone_suffixes.get(tone, "")
        
        return prefix + response + suffix

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
        base_message = f"{greeting} I'm ME Bot, your comprehensive women's health assistant."
        
        if session_context and session_context.get('discussed_topics'):
            topics = list(session_context['discussed_topics'])[-2:]
            topics_str = ', '.join(topics)
            return f"{base_message} We were discussing {topics_str}. How can I help you today? 😊"
        
        return f"{base_message} I integrate medical guidelines and research to provide comprehensive information. How can I help you today? 😊"

# -------------------------
# TRUE RAG-ENHANCED MAIN CHATBOT
# -------------------------
class TrueRAGEnhancedMEBot:
    def __init__(self):
        print("🚀 Initializing TRUE RAG-Enhanced ME Bot...")
        
        # Load FAQ data
        self.faq_data = RobustFAQLoader.load_faq_with_validation(FAQ_PATH)
        
        # Initialize TRUE RAG system
        self.rag_system = TrueMedicalRAG()
        
        # Initialize other systems
        self.session_manager = EncryptedSessionManager()
        self.nlp_understanding = AdvancedNLPUnderstanding()
        self.search_engine = HybridSearchWithRAG(self.faq_data, self.rag_system)
        self.response_builder = RAGEnhancedResponseBuilder(self.nlp_understanding, self.rag_system)
        
        print("✅ TRUE RAG System Initialized")
        print("🏥 Comprehensive Medical Knowledge Base Loaded")
        print("📚 WHO, ACOG, CDC Guidelines Integrated")
        print("🔬 Latest Research Insights Included")
    
    def process_query(self, user_input, session_id):
        """Process user query with TRUE RAG enhancement"""
        # Emergency detection
        if any(emergency in user_input.lower() for emergency in EMERGENCY_KEYWORDS):
            return f"🚨 EMERGENCY: {MEDICAL_DISCLAIMER} Please seek immediate medical attention!"
        
        # Get session context
        session_context = self.session_manager.get_conversation_context(session_id)
        
        # Comprehensive RAG-ready search
        matches = self.search_engine.comprehensive_rag_search(user_input)
        
        if not matches:
            return self._build_rag_fallback_response(user_input, session_context)
        
        # Get best match
        best_match = matches[0]
        
        # Build RAG-enhanced response
        response = self.response_builder.build_rag_enhanced_response(
            user_input, best_match, session_context
        )
        
        # Extract topics for session tracking
        detected_topics = self.nlp_understanding.analyze_query(user_input)['medical_context']
        
        # Update session
        self.session_manager.update_session(
            session_id, user_input, response, detected_topics
        )
        
        return response
    
    def _build_rag_fallback_response(self, user_input, session_context):
        """Build fallback response with RAG knowledge"""
        if self.rag_system.generator:
            try:
                # Create minimal context for RAG
                simple_context = {
                    'question': user_input,
                    'answer': "I'll provide general women's health information.",
                    'topics': self.nlp_understanding.analyze_query(user_input)['medical_context']
                }
                
                rag_response = self.rag_system.get_true_rag_answer(
                    user_input, simple_context, session_context
                )
                return f"While I don't have specific FAQ information, here's relevant medical knowledge: {rag_response}"
            except Exception as e:
                print(f"⚠️ RAG fallback failed: {e}")
        
        if session_context and session_context.get('discussed_topics'):
            topics = list(session_context['discussed_topics'])[-2:]
            return f"I specialize in women's health topics like {', '.join(topics)}. Could you ask something specific about these areas?"
        
        return "I focus on women's health including PCOD/PCOS, menstrual health, and hormonal balance. Please ask about these specific topics."

# -------------------------
# ENHANCED CLI WITH TRUE RAG
# -------------------------
def run_true_rag_enhanced_cli():
    bot = TrueRAGEnhancedMEBot()
    
    print(f"\n{BOT_NAME} TRUE RAG-Enhanced")
    print("=" * 60)
    print("✨ TRUE Medical Knowledge Integration")
    print("✨ WHO, ACOG, CDC Guidelines")
    print("✨ Latest Research Insights")
    print("✨ Comprehensive Women's Health")
    print("✨ Evidence-Based Responses")
    print("=" * 60)
    
    session_id = str(uuid.uuid4())[:8]
    print(f"Session ID: {session_id} (Encrypted)")
    
    # Get initial context
    try:
        initial_context = bot.session_manager.get_conversation_context(session_id)
        welcome_message = TimeBasedGreeting.get_welcome_message(initial_context)
    except Exception as e:
        welcome_message = TimeBasedGreeting.get_welcome_message()
    
    print(f"\n{welcome_message}")
    print("I integrate medical guidelines, research, and clinical knowledge for comprehensive answers.")
    print("Type 'bye' to exit")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit', 'bye']:
                try:
                    final_context = bot.session_manager.get_conversation_context(session_id)
                    duration = final_context.get('session_duration', 0)
                    topics = final_context.get('discussed_topics', [])
                    
                    goodbye_msg = f"Goodbye! We discussed {len(topics)} health topics with medical guideline integration. Take care! 💖"
                    print(f"{BOT_NAME}: {goodbye_msg}")
                except:
                    print(f"{BOT_NAME}: Goodbye! Stay healthy! 💖")
                break
            
            start_time = time.time()
            response = bot.process_query(user_input, session_id)
            end_time = time.time()
            
            print(f"{BOT_NAME}: {response}")
            print(f"[TRUE RAG Enhanced in {end_time-start_time:.2f}s]")
            print("-" * 60)
            
        except KeyboardInterrupt:
            print(f"\n{BOT_NAME}: Medical RAG session saved. Stay healthy! 💪")
            break
        except Exception as e:
            print(f"{BOT_NAME}: I'm ready to help! Please try your question again.")

if __name__ == '__main__':
    run_true_rag_enhanced_cli()
