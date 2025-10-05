import nltk
import json
import re
import logging
from datetime import datetime
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from fuzzywuzzy import fuzz  # For fuzzy matching
import openai  # NEW: For AI fallback
from dotenv import load_dotenv  # NEW: Load .env
import os  # NEW: For env vars

# NEW: Load .env for OpenAI key
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# ✅ Conditional NLTK downloads (your original)
required_nltk = ['punkt', 'wordnet', 'stopwords', 'omw-1.4']
for resource in required_nltk:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

# Initialize tools (your original)
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Simple abbreviation handler (your original)
abbrev_dict = {
    'ug': 'undergraduate',
    'pg': 'postgraduate',
    'bca': 'bca',
    'bcom': 'b.com',
    'nhck': 'new horizon college'
}

# Load FAQs from JSON (your original)
def load_faqs():
    try:
        with open('faqs.json', 'r', encoding='utf-8') as f:
            faqs = json.load(f)
        questions = [faq['question'] for faq in faqs]
        answers = [faq['answer'] for faq in faqs]
        return questions, answers, faqs
    except FileNotFoundError:
        print("Error: faqs.json not found. Create it with NHCK FAQs.")
        return [], [], []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in faqs.json: {e}. Check formatting.")
        return [], [], []

questions, answers, full_faqs = load_faqs()

# Preprocess text (your original, improved)
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<script\b[^<]*(?:(?!</script>)<[^<]*)*</script>', '', text)
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    for abbrev, full in abbrev_dict.items():
        text = re.sub(r'\b' + abbrev + r'\b', full, text)
    
    tokens = nltk.word_tokenize(text)
    tokens = [stemmer.stem(lemmatizer.lemmatize(word)) for word in tokens 
              if word.isalpha() and word not in stop_words]
    return " ".join(tokens)

# Preprocess all questions once (your original)
preprocessed_questions = [preprocess_text(q) for q in questions]

# Fit vectorizer once (your original, fixed)
vectorizer = None
faq_vectors = np.array([])
if questions and len(preprocessed_questions) > 0:
    try:
        vectorizer = CountVectorizer().fit(preprocessed_questions)
        faq_vectors = vectorizer.transform(preprocessed_questions).toarray()
    except Exception as e:
        print(f"Warning: Could not initialize vectorizer: {e}. Using fuzzy-only mode.")
        vectorizer = None
        faq_vectors = np.array([])

# Setup logging (your original)
logging.basicConfig(filename='chatbot_log.txt', level=logging.INFO, 
                    format='%(asctime)s - Query: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Simple context (your original)
conversation_history = []

def get_answer(user_input, context_aware=True):
    global conversation_history
    
    if not questions:
        return "FAQs not loaded. Please check faqs.json."
    
    logging.info(f":User    {user_input}")
    
    user_processed = preprocess_text(user_input)
    if not user_processed:
        return "Sorry, I didn't understand that. Try asking about NHCK courses or admissions."
    
    if context_aware and len(conversation_history) >= 1:
        context = conversation_history[-1]
        user_processed = f"{preprocess_text(context)} {user_processed}"
    
    response = None
    if vectorizer is not None:
        input_vector = vectorizer.transform([user_processed]).toarray()
        similarities = cosine_similarity(input_vector, faq_vectors)[0]
        
        best_idx = np.argmax(similarities)
        similarity_score = similarities[best_idx]
        
        if similarity_score >= 0.2:
            response = answers[best_idx]
    
    if response is None:
        fuzzy_scores = [fuzz.ratio(user_input.lower(), q.lower()) for q in questions]
        best_fuzzy_idx = np.argmax(fuzzy_scores)
        if fuzzy_scores[best_fuzzy_idx] >= 70:
            response = answers[best_fuzzy_idx]
            logging.info(f"Fuzzy match used (score: {fuzzy_scores[best_fuzzy_idx]})")
        else:
            # NEW: OpenAI Fallback (only if key set and no local match)
            if openai.api_key:
                try:
                    ai_response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are an AI assistant for New Horizon College of Karnataka (NHCK). Provide helpful, accurate advice on admissions, courses, scholarships, and college life. If unsure, suggest checking nhck.in or emailing admissions@nhck.in. Keep responses concise."},
                            {"role": "user", "content": user_input}
                        ],
                        max_tokens=200
                    )
                    response = ai_response.choices[0].message.content.strip()
                    logging.info("OpenAI fallback used.")
                except Exception as e:
                    logging.error(f"OpenAI error: {e}")
                    response = "Sorry, I couldn't generate a response right now. Try rephrasing or visit nhck.in."
            else:
                response = "Sorry, I don't have specific info on that. Try rephrasing your question about admissions, courses, or facilities. Or visit <a href='https://nhck.in/'>NHCK website</a> / email admissions@nhck.in."
    
    conversation_history.append(user_input)
    if len(conversation_history) > 6:
        conversation_history = conversation_history[-6:]
    
    logging.info(f"Response: {response}")
    return response

# Flask-compatible wrapper (your original)
def get_response(user_input):
    return get_answer(user_input)

# Console chat (your original)
if __name__ == "__main__":
    if not questions:
        print("💬 NHCK Chatbot: Error loading FAQs. Fix faqs.json and restart.")
    else:
        print("💬 NHCK Website Chatbot: Hello! I'm the New Horizon College assistant. Ask about admissions, courses, or facilities. Type 'exit' to quit.\n")
        print("Example: 'What is BCA eligibility?'\n")

        while True:
            student_input = input("You: ").strip()
            if not student_input:
                continue

            if student_input.lower() in ["exit", "quit", "bye", "goodbye"]:
                print("💬 NHCK Chatbot: Goodbye! Visit https://nhck.in/ for more.")
                break

            print("💬 NHCK Chatbot: ", end="", flush=True)
            import time; time.sleep(1)
            
            bot_response = get_answer(student_input)
            print(bot_response + "\n")