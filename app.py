

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="FAQ Chatbot",
    page_icon="🤖"
)

st.title("🤖 FAQ Chatbot")
st.write("Welcome! Ask me any question about our services.")

faq_questions = [
    "What is your name?",
    "What is a chatbot?",
    "How can I contact support?",
    "What are your working hours?",
    "How can I reset my password?",
    "What services do you provide?",
    "How can I create an account?",
    "Is this service free?"
]

faq_answers = [
    "I am an FAQ Chatbot created using Python and Streamlit.",
    "A chatbot is a software application that can communicate with users.",
    "You can contact support through our official support email.",
    "Our working hours are 9 AM to 5 PM.",
    "You can reset your password using the Forgot Password option.",
    "We provide information and answers to frequently asked questions.",
    "You can create an account by completing the registration process.",
    "Please check the service plan for pricing details."
]

vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(faq_questions)

user_question = st.text_input(
    "Enter your question:",
    placeholder="Ask me something..."
)

if st.button("Get Answer"):
    if user_question.strip():
        user_vector = vectorizer.transform([user_question])
        similarity = cosine_similarity(
            user_vector, faq_vectors
        )
        best_match = similarity.argmax()
        score = similarity[0][best_match]

        if score >= 0.15:
            st.success("Answer:")
            st.write(faq_answers[best_match])
        else:
            st.warning(
                "Sorry, I couldn't find an answer. "
                "Please try asking another question."
            )
    else:
        st.warning("Please enter a question first.")

st.markdown("---")
st.caption("Built with Python, Streamlit and Scikit-learn")
