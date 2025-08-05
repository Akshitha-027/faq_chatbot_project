# chatbot_app.py

import spacy
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Step 1: FAQ Data
faq_data = {
    "What is your return policy?": "You can return items within 30 days.",
    "How long does shipping take?": "Shipping takes 5–7 business days.",
    "Do you ship internationally?": "Yes, we ship worldwide.",
    "How can I track my order?": "Use the tracking link in your confirmation email.",
    "What payment methods are accepted?": "We accept cards, UPI, and PayPal.",
    "Can I cancel my order?": "Yes, you can cancel your order within 24 hours of placing it.",
    "Do you offer customer support?": "Yes, we offer 24/7 customer support through chat and email."
}

# Extract questions and answers
questions = list(faq_data.keys())
answers = list(faq_data.values())

# ✅ Step 2: Preprocess Questions
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    doc = nlp(text.lower())
    return " ".join(token.lemma_ for token in doc if token.is_alpha and not token.is_stop)

preprocessed_questions = [preprocess(q) for q in questions]

# ✅ Step 3: TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_questions)

# ✅ Step 4: Match User Input
def get_best_answer(user_input):
    processed_input = preprocess(user_input)
    user_vec = vectorizer.transform([processed_input])
    similarity = cosine_similarity(user_vec, tfidf_matrix)
    best_index = similarity.argmax()
    return answers[best_index]

# ✅ Step 5: Streamlit Chat UI
st.set_page_config(page_title="FAQ Chatbot", page_icon="💬")
st.title("💬 FAQ Chatbot")
st.write("Ask me anything about shipping, returns, or payments!")

user_question = st.text_input("You:")

if user_question:
    response = get_best_answer(user_question)
    st.markdown(f"**Bot:** {response}")
