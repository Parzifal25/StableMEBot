import os
import json
import time
import numpy as np
from datetime import datetime
import gc
from functools import lru_cache

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
FAQ_PATH = os.path.join(BASE_DIR, 'faq.json')
EMB_PATH = os.path.join(BASE_DIR, 'faq_embeddings.npy')
EMB_META = os.path.join(BASE_DIR, 'faq_emb_meta.json')
CONTEXT_FILE = os.path.join(BASE_DIR, 'context_memory.json')

BOT_NAME = "ME Bot"
SEM_THRESHOLD = 0.68
FUZZY_THRESHOLD = 82
PHONETIC_THRESHOLD = 1
MAX_CONTEXT = 8
SAFE_FALLBACK_MAX_TOKENS = 60
DOMAIN_KEYWORDS = ['pcod', 'pcos', 'pregnancy', 'postpartum', 'ovary', 'ovaries', 'hormone', 'menstrual']

# Medical safety
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

tfidf_vectorizer = None
tfidf_matrix = None

# Response cache for performance
RESPONSE_CACHE = {}
MAX_CACHE_SIZE = 1000

# Conversation tracking
CONVERSATION_STAGES = {
    'greeting': 0,
    'query': 1, 
    'follow_up': 2,
    'explanation': 3,
    'closing': 4
}

# -------------------------
# Lazy Loading System for Low-spec Optimization
# -------------------------
class LazyFAQSystem:
    def __init__(self, faq_path):
        self.faq_path = faq_path
        self._embeddings = None
        self._faq_data = None
        self._faq_questions = None
        self._faq_answers = None
        self._faq_phonetic = None
    
    @property
    def faq_data(self):
        if self._faq_data is None:
            with open(self.faq_path, 'r', encoding='utf-8') as f:
                self._faq_data = json.load(f)
        return self._faq_data
    
    @property 
    def faq_questions(self):
        if self._faq_questions is None:
            self._faq_questions = [q['question'].strip() for q in self.faq_data]
        return self._faq_questions
    
    @property
    def faq_answers(self):
        if self._faq_answers is None:
            self._faq_answers = [q['answer'].strip() for q in self.faq_data]
        return self._faq_answers

# Initialize lazy system
lazy_faq = LazyFAQSystem(FAQ_PATH)

# Load FAQ data
FAQ = lazy_faq.faq_data
FAQ_QUESTIONS = lazy_faq.faq_questions
FAQ_ANSWERS = lazy_faq.faq_answers

# -------------------------
# Precompute or load embeddings with chunking for large data
# -------------------------
def process_large_faq_in_chunks(faq_questions, chunk_size=50):
    """Process FAQ in chunks to avoid memory overload"""
    print(f"Computing embeddings in chunks of {chunk_size}...")
    all_embeddings = []
    
    for i in range(0, len(faq_questions), chunk_size):
        chunk = faq_questions[i:i + chunk_size]
        chunk_embeddings = embed_model.encode(chunk, convert_to_numpy=True, show_progress_bar=False)
        all_embeddings.extend(chunk_embeddings)
        # Force garbage collection
        gc.collect()
    
    return np.array(all_embeddings)

if os.path.exists(EMB_PATH) and os.path.exists(EMB_META):
    try:
        faq_embeddings = np.load(EMB_PATH)
        with open(EMB_META, 'r', encoding='utf-8') as m:
            meta = json.load(m)
        if len(meta.get('questions', [])) != len(FAQ_QUESTIONS):
            raise Exception("FAQ changed, will recompute embeddings.")
    except Exception:
        faq_embeddings = None
else:
    faq_embeddings = None

if faq_embeddings is None:
    print("Computing FAQ embeddings (optimized for low-spec)...")
    if len(FAQ_QUESTIONS) > 100:
        faq_embeddings = process_large_faq_in_chunks(FAQ_QUESTIONS)
    else:
        faq_embeddings = embed_model.encode(FAQ_QUESTIONS, convert_to_numpy=True, show_progress_bar=True)
    
    np.save(EMB_PATH, faq_embeddings)
    with open(EMB_META, 'w', encoding='utf-8') as m:
        json.dump({'questions': FAQ_QUESTIONS}, m)

# Build FAISS index if available
if USE_FAISS:
    d = faq_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    norms = np.linalg.norm(faq_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    faq_emb_norm = faq_embeddings / norms
    index.add(faq_emb_norm.astype('float32'))
else:
    normalized_qs = []
    for q in FAQ_QUESTIONS:
        tokens = word_tokenize(q.lower())
        tokens = [LEMMA.lemmatize(t) for t in tokens if t.isalpha() and t not in STOPWORDS]
        normalized_qs.append(" ".join(tokens))
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(normalized_qs)
    # Optimize memory by converting to sparse format
    tfidf_matrix = tfidf_matrix.tocsr()

# -------------------------
# Precompute phonetic codes
# -------------------------
FAQ_PHONETIC = []
for q in FAQ_QUESTIONS:
    tokens = [t for t in word_tokenize(q.lower()) if t.isalpha()]
    codes = [phonetics.dmetaphone(t) for t in tokens]
    code_set = set([c for pair in codes for c in pair if c])
    FAQ_PHONETIC.append(code_set)

# -------------------------
# GPT2 fallback (restricted)
# -------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpt_tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
gpt_model = GPT2LMHeadModel.from_pretrained('distilgpt2').to(device)
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

# -------------------------
# Context memory
# -------------------------
if os.path.exists(CONTEXT_FILE):
    try:
        with open(CONTEXT_FILE, 'r', encoding='utf-8') as cf:
            CONTEXT_MEMORY = json.load(cf)
    except Exception:
        CONTEXT_MEMORY = []
else:
    CONTEXT_MEMORY = []

def persist_context():
    try:
        with open(CONTEXT_FILE, 'w', encoding='utf-8') as cf:
            json.dump(CONTEXT_MEMORY[-500:], cf, indent=2)
    except Exception:
        pass

# -------------------------
# INTELLIGENT FAQ RESPONSE GENERATION SYSTEM
# -------------------------
def simplify_medical_text(text):
    """
    Simplify complex medical jargon for better understanding
    """
    simplifications = {
        'hormonal imbalance': 'hormone levels that are not balanced properly',
        'ovulatory dysfunction': 'problems with releasing eggs from ovaries',
        'metabolic abnormalities': 'issues with how your body processes energy',
        'insulin resistance': 'when your body doesn\'t respond well to insulin hormone',
        'hirsutism': 'unwanted hair growth in women',
        'amenorrhea': 'missing periods regularly',
        'endocrine disorder': 'health condition affecting your hormones',
        'polycystic ovaries': 'ovaries with many small follicles',
        'androgen levels': 'male hormone levels in your body',
        'fertility issues': 'difficulty in getting pregnant',
        'inflammation': 'body\'s response to irritation or injury',
        'metabolic syndrome': 'group of conditions that increase health risks'
    }
    
    simple_text = text
    for complex_term, simple_explanation in simplifications.items():
        if complex_term.lower() in simple_text.lower():
            simple_text = simple_text.replace(complex_term, simple_explanation)
            # Also handle capitalized versions
            simple_text = simple_text.replace(complex_term.title(), simple_explanation)
    
    return simple_text

def detect_user_emotion(query):
    """
    Detect user's emotional state from query
    """
    query_lower = query.lower()
    
    emotional_cues = {
        'worried': ['worried', 'anxious', 'scared', 'nervous', 'afraid', 'stress', 'stressed'],
        'confused': ['confused', 'dont understand', 'explain', 'what does', 'mean', 'clarify'],
        'seeking_help': ['help', 'advice', 'suggestion', 'what should i do', 'how to', 'treatment'],
        'frustrated': ['frustrated', 'annoying', 'tired of', 'sick of', 'had enough'],
        'information': ['what is', 'tell me about', 'information about', 'explain'],
        'curious': ['curious', 'wonder', 'interested', 'want to know']
    }
    
    for emotion, cues in emotional_cues.items():
        if any(cue in query_lower for cue in cues):
            return emotion
    
    return 'neutral'

def generate_empathetic_response(faq_answer, user_query, tone, needs_simplification=False):
    """
    Transform FAQ answers into natural, tone-appropriate responses
    """
    user_emotion = detect_user_emotion(user_query)
    
    # Base response templates by tone and emotion
    templates = {
        'empathic': {
            'worried': "I understand this might be worrying. {answer} Remember, many women experience this and there are effective ways to manage it. You're not alone in this.",
            'confused': "Let me clarify this for you: {answer} Does this make sense? Feel free to ask more questions if anything is unclear.",
            'seeking_help': "I'm here to help you. Based on available information: {answer} However, please consult a doctor for personalized medical advice tailored to your situation.",
            'frustrated': "I hear your frustration. Dealing with health issues can be challenging. {answer} Many women find that with proper care, things do get better.",
            'information': "I appreciate you seeking information about this. {answer}",
            'curious': "That's a great question! {answer} Learning about your health is an important step.",
            'neutral': "Thank you for your question. {answer}"
        },
        'friendly': {
            'worried': "Hey there! 😊 No need to worry. {answer} Many women go through this and come out stronger! Remember, you've got this!",
            'confused': "No worries! Let me break this down for you: {answer} Hope that helps! Ask me anything else if you're still curious!",
            'seeking_help': "Sure, I'd love to help! {answer} Remember, I'm here for general info - your doctor knows best for personalized advice!",
            'frustrated': "I get it, this can be really frustrating sometimes! 😔 {answer} Hang in there - taking small steps can make a big difference!",
            'information': "Great question! 😄 {answer}",
            'curious': "Ooh, interesting question! {answer} Keep those questions coming!",
            'neutral': "Hi! {answer} 😊"
        },
        'informative': {
            'worried': "Clinical perspective: {answer} It's important to note that professional medical consultation is recommended for proper diagnosis and treatment.",
            'confused': "Let me provide a clear explanation: {answer} Key points to remember: This condition varies among individuals and professional guidance is essential.",
            'seeking_help': "Based on medical literature: {answer} For personalized treatment plans, healthcare provider consultation is strongly recommended.",
            'frustrated': "I understand this can be challenging. From a medical standpoint: {answer} Consistent care and professional guidance often lead to improvement.",
            'information': "{answer}",
            'curious': "Research indicates: {answer} Continued learning about health conditions is valuable for self-advocacy.",
            'neutral': "{answer}"
        }
    }
    
    # Simplify the answer if needed
    if needs_simplification:
        processed_answer = simplify_medical_text(faq_answer)
    else:
        processed_answer = faq_answer
    
    # Get appropriate template
    tone_templates = templates.get(tone, templates['informative'])
    response_template = tone_templates.get(user_emotion, "{answer}")
    
    final_response = response_template.format(answer=processed_answer)
    return final_response

# -------------------------
# Utility functions with caching
# -------------------------
@lru_cache(maxsize=1000)
def cached_normalize(text):
    """Cached normalization for performance"""
    text = text.strip()
    tokens = word_tokenize(text.lower())
    tokens = [LEMMA.lemmatize(t) for t in tokens if t.isalpha()]
    return " ".join(tokens)

def spell_correct(text):
    words = word_tokenize(text)
    corrected = []
    for w in words:
        if not w.isalpha():
            corrected.append(w)
            continue
        if w.lower() in SPELL:
            corrected.append(w)
        else:
            c = SPELL.correction(w)
            corrected.append(c if c else w)
    return " ".join(corrected)

def normalize(text):
    text = text.strip()
    text = spell_correct(text)
    return cached_normalize(text)

def phonetic_codes(text):
    tokens = [t for t in word_tokenize(text.lower()) if t.isalpha()]
    codes = [phonetics.dmetaphone(t) for t in tokens]
    code_set = set([c for pair in codes for c in pair if c])
    return code_set

@lru_cache(maxsize=500)
def cached_semantic_search(query, top_k=1):
    """Cached semantic search"""
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    q_norm = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
    if USE_FAISS:
        D, I = index.search(q_norm.astype('float32'), top_k)
        return I[0].tolist(), D[0].tolist()
    else:
        q_vec = tfidf_vectorizer.transform([query])
        sims = cosine_similarity(q_vec, tfidf_matrix)[0]
        idxs = sims.argsort()[::-1][:top_k]
        return idxs.tolist(), [float(sims[i]) for i in idxs]

def semantic_search(query, top_k=1):
    return cached_semantic_search(query, top_k)

def fuzzy_search(query):
    scores = [fuzz.token_set_ratio(query.lower(), q.lower()) for q in FAQ_QUESTIONS]
    best_idx = int(np.argmax(scores))
    return best_idx, scores[best_idx]

def tone_for_query(query):
    q = query.lower()
    if any(g in q for g in ['hello','hi','hey','good morning','good evening']):
        return 'friendly'
    if any(w in q for w in ['pain','bleeding','severe','emergency','urgent', 'worried', 'scared']):
        return 'empathic'
    return 'informative'

def apply_tone(text, tone):
    if tone == 'friendly':
        return f"{BOT_NAME}: {text} 😊"
    if tone == 'empathic':
        return f"{BOT_NAME}: {text} 💙"
    return f"{BOT_NAME}: {text}"

# -------------------------
# Response caching system
# -------------------------
def get_cached_answer(query):
    """Get cached response or compute new one"""
    normalized = normalize(query)
    
    if normalized in RESPONSE_CACHE:
        return RESPONSE_CACHE[normalized]
    
    # Compute new answer
    answer = get_answer_impl(query)
    RESPONSE_CACHE[normalized] = answer
    
    # Limit cache size
    if len(RESPONSE_CACHE) > MAX_CACHE_SIZE:
        # Remove oldest entry (Python 3.7+ preserves insertion order)
        first_key = next(iter(RESPONSE_CACHE))
        del RESPONSE_CACHE[first_key]
    
    return answer

# -------------------------
# Emergency detection
# -------------------------
def detect_emergency(query):
    """Check if query indicates medical emergency"""
    query_lower = query.lower()
    for emergency_word in EMERGENCY_KEYWORDS:
        if emergency_word in query_lower:
            return True
    return False

# -------------------------
# Dynamic greeting message
# -------------------------
def get_greeting():
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "Good morning! ☀️ I'm ME Bot, your personal health assistant. How can I help you today?"
    elif 12 <= hour < 17:
        return "Good afternoon! 🌤️ I'm ME Bot, here to assist you with your queries."
    elif 17 <= hour < 22:
        return "Good evening! 🌙 ME Bot at your service. How are you feeling today?"
    else:
        return "Burning the midnight oil? 🌌 I'm ME Bot — still awake and ready to chat!"

# -------------------------
# Core responder logic
# -------------------------
def get_answer_impl(user_input):
    # Emergency detection
    if detect_emergency(user_input):
        return f"🚨 EMERGENCY: {MEDICAL_DISCLAIMER} Please contact emergency services or visit the nearest hospital immediately."
    
    corrected = spell_correct(user_input)
    normalized_query = normalize(user_input)
    phonetic_q = phonetic_codes(user_input)

    low = user_input.strip().lower()
    if any(low == w for w in ['hello','hiiiiiiiiiiiii','hey','namaste','vanakam']):
        return apply_tone("Hello! I'm ME Bot, your health assistant. How can I help?", 'friendly')
    if low in ['bye','goodbye','exit']:
        return apply_tone("Goodbye! Take care of yourself! 💖", 'friendly')

    # Context-aware responses
    if CONTEXT_MEMORY:
        last_bot = CONTEXT_MEMORY[-1]['bot'] if CONTEXT_MEMORY else ""
        if 'similar' in user_input.lower() or 'difference' in user_input.lower():
            if ('PCOD' in last_bot or 'PCOS' in last_bot) or any(term in last_bot.lower() for term in ['pcod','pcos']):
                return apply_tone("PCOD and PCOS are related but distinct. PCOD involves multiple follicles in ovaries; PCOS is a hormonal disorder affecting ovulation. Both need medical supervision.", 'informative')

    # Check if user wants simpler explanation
    needs_simple = any(word in user_input.lower() for word in 
                      ['explain simply', 'in simple terms', 'easy explanation', 'dumb it down', 'like i\'m 5'])

    # Multi-layer search
    ids, scores = semantic_search(normalized_query, top_k=3)
    best_id = ids[0]
    sem_score = float(scores[0])
    
    if sem_score >= SEM_THRESHOLD:
        base_answer = FAQ_ANSWERS[best_id]
        tone = tone_for_query(user_input)
        
        # Generate empathetic response
        empathetic_answer = generate_empathetic_response(
            base_answer, user_input, tone, needs_simplification=needs_simple
        )
        
        CONTEXT_MEMORY.append({'user': user_input, 'bot': empathetic_answer, 'timestamp': datetime.now().isoformat()})
        if len(CONTEXT_MEMORY) > MAX_CONTEXT: 
            CONTEXT_MEMORY.pop(0)
        persist_context()
        return apply_tone(empathetic_answer, tone)

    # Phonetic matching
    for i, code_set in enumerate(FAQ_PHONETIC):
        if len(phonetic_q & code_set) >= PHONETIC_THRESHOLD:
            base_answer = FAQ_ANSWERS[i]
            tone = tone_for_query(user_input)
            empathetic_answer = generate_empathetic_response(
                base_answer, user_input, tone, needs_simplification=needs_simple
            )
            CONTEXT_MEMORY.append({'user': user_input, 'bot': empathetic_answer, 'timestamp': datetime.now().isoformat()})
            if len(CONTEXT_MEMORY) > MAX_CONTEXT: 
                CONTEXT_MEMORY.pop(0)
            persist_context()
            return apply_tone(empathetic_answer, tone)

    # Fuzzy matching
    fidx, fscore = fuzzy_search(user_input)
    if fscore >= FUZZY_THRESHOLD:
        base_answer = FAQ_ANSWERS[fidx]
        tone = tone_for_query(user_input)
        empathetic_answer = generate_empathetic_response(
            base_answer, user_input, tone, needs_simplification=needs_simple
        )
        CONTEXT_MEMORY.append({'user': user_input, 'bot': empathetic_answer, 'timestamp': datetime.now().isoformat()})
        if len(CONTEXT_MEMORY) > MAX_CONTEXT: 
            CONTEXT_MEMORY.pop(0)
        persist_context()
        return apply_tone(empathetic_answer, tone)

    # GPT-2 fallback for non-domain queries
    if not any(k in normalized_query for k in DOMAIN_KEYWORDS):
        prompt_parts = []
        for e in CONTEXT_MEMORY[-4:]:  # Reduced context for performance
            prompt_parts.append(f"User: {e['user']}\nBot: {e['bot']}")
        
        healthcare_prompt = f"""You are a women's health assistant. Provide general, supportive information only.
        
Recent conversation:
{" ".join(prompt_parts)}

User: {user_input}
Assistant: I can provide general information about women's health. For medical concerns, please consult a doctor. Based on general knowledge:"""

        inputs = gpt_tokenizer(healthcare_prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = gpt_model.generate(
                **inputs,
                max_length=min(inputs['input_ids'].shape[1] + SAFE_FALLBACK_MAX_TOKENS, 200),
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                pad_token_id=gpt_tokenizer.pad_token_id,
                early_stopping=True
            )
        text = gpt_tokenizer.decode(out[0], skip_special_tokens=True)
        response = text.split('Assistant:')[-1].strip() if 'Assistant:' in text else text.split('Bot:')[-1].strip() if 'Bot:' in text else text.strip()
        
        # Safety filter
        if len(response.split()) > 70 or any(bad in response.lower() for bad in ['http', 'www', '.com', 'prescription', 'diagnose']):
            response = f"Sorry, I don't have a precise answer to that. I can help with PCOD/PCOS, postpartum care, pregnancy, and related women's health topics. For medical advice, please consult a healthcare professional."
        
        CONTEXT_MEMORY.append({'user': user_input, 'bot': response, 'timestamp': datetime.now().isoformat()})
        if len(CONTEXT_MEMORY) > MAX_CONTEXT: 
            CONTEXT_MEMORY.pop(0)
        persist_context()
        return apply_tone(response, tone_for_query(user_input))

    # Default response for domain queries without matches
    default_resp = f"Sorry, I don't have specific information on that yet. I'm constantly learning about PCOD, PCOS, pregnancy, and postpartum care. For medical concerns please consult a healthcare professional."
    CONTEXT_MEMORY.append({'user': user_input, 'bot': default_resp, 'timestamp': datetime.now().isoformat()})
    if len(CONTEXT_MEMORY) > MAX_CONTEXT: 
        CONTEXT_MEMORY.pop(0)
    persist_context()
    return apply_tone(default_resp, tone_for_query(user_input))

def get_answer(user_input):
    """Public interface with caching and error handling"""
    try:
        # Input validation
        if len(user_input.strip()) < 2:
            return "Please provide more details about your question."
        
        if len(user_input) > 500:
            return "Your question is a bit long. Could you please summarize your main concern?"
            
        return get_cached_answer(user_input)
    except Exception as e:
        print(f"Error processing query: {e}")
        return "I'm experiencing some technical difficulties. Please try again or rephrase your question. For urgent matters, please consult a healthcare professional directly."

# -------------------------
# Memory optimization function
# -------------------------
def optimize_memory():
    """Call this periodically to free up memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# -------------------------
# CLI + Flask API
# -------------------------
app = Flask(__name__)
CORS(app)

@app.route('/api/chat', methods=['POST'])
def api_chat():
    body = request.json or {}
    user_input = body.get('message', '').strip()
    if not user_input:
        return jsonify({'response': f"{BOT_NAME}: Please send a non-empty message."})
    
    resp = get_answer(user_input)
    return jsonify({'response': resp})

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'bot_name': BOT_NAME,
        'faq_count': len(FAQ_QUESTIONS),
        'cache_size': len(RESPONSE_CACHE),
        'context_memory': len(CONTEXT_MEMORY)
    })

def run_cli():
    print(f"{BOT_NAME} running (CLI) - Optimized for Low-spec")
    print(get_greeting())
    print("\nTips: You can ask me to 'explain simply' or 'in simple terms' for easier explanations!")
    
    query_count = 0
    while True:
        try:
            q = input("You: ").strip()
            if q.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print(f"{BOT_NAME}: Goodbye! Take care of yourself! 💖")
                break
            if q.lower() == 'clear cache':
                RESPONSE_CACHE.clear()
                print(f"{BOT_NAME}: Cache cleared! 🧹")
                continue
                
            start_time = time.time()
            response = get_answer(q)
            end_time = time.time()
            
            print(response)
            print(f"[Processed in {end_time-start_time:.2f}s]")
            
            query_count += 1
            # Optimize memory every 10 queries
            if query_count % 10 == 0:
                optimize_memory()
                
        except KeyboardInterrupt:
            print(f"\n{BOT_NAME}: Session ended. Stay healthy! 💪")
            break
        except Exception as e:
            print(f"{BOT_NAME}: Sorry, I encountered an error. Please try again.")

if __name__ == '__main__':
    import sys
    print(f"=== {BOT_NAME} - Women's Health Assistant ===")
    print(f"FAQ Database: {len(FAQ_QUESTIONS)} questions")
    print(f"Using {'FAISS' if USE_FAISS else 'TFIDF'} for semantic search")
    print(f"Device: {device}")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == '--cli':
        run_cli()
    else:
        print(f"{BOT_NAME} Phase-2 starting API on port 5000")
        print("Endpoints:")
        print("  POST /api/chat - Chat endpoint")
        print("  GET  /api/health - Health check")
        app.run(port=5000, debug=False)