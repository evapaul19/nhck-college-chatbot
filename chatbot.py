import json
import re
import logging
import os
from dotenv import load_dotenv
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from fuzzywuzzy import fuzz
from openai import OpenAI  # v1.x client - fixes deprecation error

# Load .env for OpenAI key (secure for local/Render)
load_dotenv()
openai_client = None
api_key = os.getenv('OPENAI_API_KEY')
if api_key:
    openai_client = OpenAI(api_key=api_key)
    logging.info("OpenAI client initialized successfully.")

# NLTK Downloads: Runtime check and download (fixes Render 'punkt_tab' error)
required_nltk = ['punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']  # punkt_tab for NLTK 3.8+ / Python 3.13
for resource in required_nltk:
    try:
        if resource == 'punkt_tab':
            nltk.data.find('tokenizers/punkt_tab/english.pickle')  # Specific check for English tokenizer
        else:
            nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)
        logging.info(f"Downloaded NLTK resource: {resource}")

# Initialize tools (after NLTK data is ensured)
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Enhanced abbreviation handler (fixes typos like 'schloarships' -> scholarships)
abbrev_dict = {
    'ug': 'undergraduate',
    'pg': 'postgraduate',
    'bca': 'bachelor of computer applications',
    'bcom': 'bachelor of commerce',
    'bsc': 'bachelor of science',
    'mtech': 'master of technology',
    'mba': 'master of business administration',
    'nhck': 'new horizon college of karnataka',
    'admission': 'admissions process',
    'doc': 'documents',
    'sch': 'scholarships',
    'loc': 'location',
    'fee': 'fees',
    'lamguages': 'languages',  # From your log typos
    'prograamming': 'programming',
    'languageds': 'languages',
    'schloarship': 'scholarships',
    'schlarships': 'scholarships'
}

# Global variables (lazy-init to prevent startup crashes)
questions = []
answers = []
full_faqs = []
vectorizer = None
faq_vectors = np.array([])
preprocessed_questions = []

# Isolated logging setup (chatbot-only; no Flask pollution or dev warnings)
logger = logging.getLogger('NHCKChatbot')
logger.setLevel(logging.INFO)
handler = logging.FileHandler('chatbot_log.txt')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
if handler not in logger.handlers:  # Avoid duplicate handlers
    logger.addHandler(handler)
logger.info("NHCK Chatbot module loaded successfully.")

# Load FAQs from JSON (strict validation - fixes duplicates/mismatches)
def load_faqs():
    global questions, answers, full_faqs
    try:
        with open('faqs.json', 'r', encoding='utf-8') as f:
            full_faqs = json.load(f)
        if not isinstance(full_faqs, list) or len(full_faqs) == 0:
            raise ValueError("faqs.json must be a non-empty list of FAQ objects.")
        # Filter valid entries and extract
        valid_faqs = [faq for faq in full_faqs if isinstance(faq, dict) and 'question' in faq and 'answer' in faq]
        if len(valid_faqs) != len(full_faqs):
            logger.warning(f"Skipped {len(full_faqs) - len(valid_faqs)} invalid FAQ entries.")
        questions = [faq['question'].strip() for faq in valid_faqs]
        answers = [faq['answer'].strip() for faq in valid_faqs]
        if len(questions) != len(answers) or len(questions) == 0:
            raise ValueError("No valid question-answer pairs found.")
        # Deduplicate questions (case-insensitive)
        unique_faqs = []
        seen = set()
        for q, a in zip(questions, answers):
            q_lower = q.lower()
            if q_lower not in seen:
                seen.add(q_lower)
                unique_faqs.append({'question': q, 'answer': a})
        questions = [faq['question'] for faq in unique_faqs]
        answers = [faq['answer'] for faq in unique_faqs]
        full_faqs = unique_faqs
        logger.info(f"Loaded {len(questions)} unique FAQs from faqs.json.")
        return True
    except FileNotFoundError:
        logger.error("faqs.json not found. Create it with NHCK FAQs (e.g., questions about BCA, admissions).")
        return False
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.error(f"Invalid faqs.json format: {e}. Ensure valid JSON with 'question' and 'answer' keys, no duplicates.")
        return False
    except Exception as e:
        logger.error(f"Unexpected error loading FAQs: {e}")
        return False

# Preprocess text (robust - handles errors, typos via abbrevs)
def preprocess_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    # Clean HTML/scripts and normalize
    text = re.sub(r'<script\b[^<]*(?:(?!</script>)<[^<]*)*</script>', '', text)
    text = re.sub(r'[^\w\s]', '', text.lower().strip())
    # Replace abbreviations (regex-safe)
    for abbrev, full in abbrev_dict.items():
        text = re.sub(r'\b' + re.escape(abbrev) + r'\b', full, text)
    # Tokenize, filter, lemmatize/stem
    try:
        tokens = nltk.word_tokenize(text)
        filtered_tokens = [
            stemmer.stem(lemmatizer.lemmatize(word))
            for word in tokens
            if word.isalpha() and len(word) > 2 and word not in stop_words
        ]
        return " ".join(filtered_tokens)
    except Exception as e:
        logger.warning(f"Preprocessing error for '{text[:50]}...': {e}")
        return ""

# Lazy preprocess and vectorize (runs on first query - fixes startup crashes)
def initialize_vectorizer():
    global preprocessed_questions, vectorizer, faq_vectors
    if not questions or len(questions) == 0:
        logger.warning("No FAQs to vectorize.")
        return False
    preprocessed_questions = [preprocess_text(q) for q in questions]
    non_empty = [p for p in preprocessed_questions if p.strip()]
    if len(non_empty) > 0:
        try:
            vectorizer = CountVectorizer(min_df=1, stop_words='english').fit(non_empty)
            faq_vectors = vectorizer.transform(preprocessed_questions).toarray()
            logger.info(f"Vectorizer initialized with {len(non_empty)} non-empty FAQs.")
            return True
        except Exception as e:
            logger.warning(f"Vectorizer init failed: {e}. Using fuzzy-only mode.")
            vectorizer = None
            faq_vectors = np.array([])
            return False
    logger.warning("All preprocessed questions empty; using fuzzy-only.")
    return False

# Initial safe load (no vectorizer yet - lazy)
load_success = load_faqs()

# Conversation history (limited for context)
conversation_history = []

def get_answer(user_input, context_aware=True):
    global conversation_history, load_success
    
    if not load_success or not questions:
        return "FAQs not loaded. Please check faqs.json and restart."
    
    if not user_input or not isinstance(user_input, str):
        default = "Sorry, I didn't understand that. Try asking about NHCK courses, admissions, or scholarships."
        logger.info(f"Bot response: {default}")
        return default
    
    logger.info(f"User  query: {user_input}")
    
    user_processed = preprocess_text(user_input)
    if not user_processed:
        default = "Sorry, I didn't understand that. Try rephrasing your question about NHCK."
        logger.info(f"Bot response: {default}")
        return default
    
    # Context (prepend last query if enabled)
    if context_aware and conversation_history:
        context = preprocess_text(conversation_history[-1])
        if context:
            user_processed = f"{context} {user_processed}"
    
    response = None
    
    # Cosine similarity (strict 0.3 threshold - fixes mismatches like admission -> BCA)
    if vectorizer is not None and len(faq_vectors) > 0:
        try:
            input_vector = vectorizer.transform([user_processed]).toarray()
            similarities = cosine_similarity(input_vector, faq_vectors)[0]
            best_idx = np.argmax(similarities)
            if similarities[best_idx] >= 0.3:  # Tuned higher for precision
                response = answers[best_idx]
                logger.info(f"Cosine match (score: {similarities[best_idx]:.2f}, FAQ #{best_idx})")
        except Exception as e:
            logger.warning(f"Cosine error: {e}")
    
    # Fuzzy fallback (strict 80% threshold - fixes weak matches)
    if response is None:
        try:
            fuzzy_scores = [fuzz.ratio(user_input.lower(), q.lower()) for q in questions]
            best_fuzzy_idx = np.argmax(fuzzy_scores)
            if fuzzy_scores[best_fuzzy_idx] >= 80:  # Tuned higher for accuracy
                response = answers[best_fuzzy_idx]
                logger.info(f"Fuzzy match (score: {fuzzy_scores[best_fuzzy_idx]}, FAQ #{best_fuzzy_idx})")
        except Exception as e:
            logger.warning(f"Fuzzy error: {e}")
    
    # OpenAI fallback (v1.x API - fixes deprecation; NHCK-specific prompt)
    if response is None and openai_client:
        try:
            completion = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI assistant for New Horizon College of Karnataka (NHCK). Provide helpful, accurate, concise advice (under 150 words) on admissions, courses (e.g., BCA), scholarships, facilities, and college life. If unsure, suggest nhck.in or admissions@nhck.in. Be friendly."
                    },
                    {"role": "user", "content": user_input}
                ],
                max_tokens=200,
                temperature=0.7
            )
            response = completion.choices[0].message.content.strip()
            logger.info("OpenAI fallback used.")
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            response = "Sorry, temporary issue with AI response. Try rephrasing or visit nhck.in."
    
    # Default no-match
    if response is None:
        response = "Sorry, I don't have info on that yet. Try rephrasing about NHCK admissions, BCA, scholarships, or location. Visit <a href='https://nhck.in/'>nhck.in</a> or email admissions@nhck.in."
    
    # Update history (limit 5)
    conversation_history.append(user_input)
    if len(conversation_history) > 5:
        conversation_history = conversation_history[-5:]
    
    # Log truncated response
    log_resp = response[:100] + "..." if len(response) > 100 else response
    logger.info(f"Bot response: {log_resp}")
    
    return response

# Flask wrapper (lazy init vectorizer)
def get_response(user_input):
    global load_success
    if not load_success:
        load_success = load_faqs()
    if load_success and questions and vectorizer is None:
        initialize_vectorizer()
    return get_answer(user_input)

# Console mode (local testing only)
if __name__ == "__main__":
    if not load_success:
        print("💬 NHCK Chatbot: Error loading FAQs. Fix faqs.json.")
    else:
        print("💬 NHCK Chatbot: Hello! Ask about NHCK. Type 'exit' to quit.\nExample: 'What documents for admission?'")
        import time
        while True:
            student_input = input("You: ").strip()
            if not student_input:
                continue
            if student_input.lower() in ["exit", "quit", "bye"]:
                print("💬 NHCK Chatbot: Goodbye! Visit nhck.in.")
                break
            print("💬 NHCK Chatbot: ", end="", flush=True)
            time.sleep(1)
            bot_response = get_response(student_input)
            print(bot_response + "\n")