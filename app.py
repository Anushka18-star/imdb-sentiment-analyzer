import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page Configuration
st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>

/* Main App Background */
.stApp {
    background: linear-gradient(to right, #dfe9f3, #ffffff);
}

/* Movie Name Input */
.stTextInput input {
    background-color: #E6E6FA;
    color: black;
    border-radius: 15px;
    border: 2px solid #9370DB;
    font-size: 18px;
    font-weight: 500;
    padding: 12px;
}

/* Review Text Area */
.stTextArea textarea {
    background-color: #E6E6FA;
    color: black;
    border-radius: 18px;
    border: 2px solid #9370DB;
    font-size: 18px;
    font-weight: 500;
    padding: 15px;
}

/* Labels */
label {
    color: black !important;
    font-size: 24px !important;
    font-weight: bold !important;
    font-family: "Times New Roman", serif;
}

/* Button Styling */
.stButton button {
    width: 100%;
    border-radius: 15px;
    height: 3.2em;
    background: linear-gradient(to right, #8e2de2, #4a00e0);
    color: white;
    font-size: 22px;
    font-weight: bold;
    border: none;
    transition: 0.3s;
}

.stButton button:hover {
    transform: scale(1.02);
}

/* Result Box */
.result-box {
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    margin-top: 30px;
    background-color: #E6E6FA;
    border: 2px solid #9370DB;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
}

/* Sidebar Cards */
.feature-card {
    background-color: #E6E6FA;
    padding: 25px;
    border-radius: 20px;
    margin-top: 20px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
}

/* Footer */
.footer {
    text-align: center;
    color: black;
    margin-top: 50px;
    font-size: 18px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# MAIN TITLE
st.markdown("""
<h1 style='
text-align: center;
font-size: 72px;
font-weight: bold;
font-family: "Times New Roman", serif;
color: black;
margin-top: -20px;
letter-spacing: 2px;
'>
🎬 AI Sentiment Analyzer 🎬
</h1>
""", unsafe_allow_html=True)

# SUBTITLE
st.markdown("""
<h3 style='
text-align: center;
font-size: 34px;
font-family: "Times New Roman", serif;
font-weight: bold;
color: black;
margin-top: 10px;
margin-bottom: 40px;
'>
✨ Analyze IMDb Movie Reviews Using Machine Learning & NLP ✨
</h3>
""", unsafe_allow_html=True)

# Layout
col1, col2 = st.columns([2, 1])

# LEFT COLUMN
with col1:

    # Movie Name
    movie_name = st.text_input(
        "🎥 Enter Movie Name",
        placeholder="Type movie name here..."
    )

    # Review Input
    user_input = st.text_area(
        "📝 Enter Movie Review",
        height=250,
        placeholder="Type your movie review here..."
    )

    # Analyze Button
    if st.button("🚀 Analyze Sentiment"):

        if user_input.strip() != "":

            # Transform text
            transformed_input = vectorizer.transform([user_input])

            # Prediction
            prediction = model.predict(transformed_input)[0]

            # Confidence
            probability = model.predict_proba(transformed_input)
            confidence = max(probability[0]) * 100

            # POSITIVE RESULT
            if prediction == "positive":

                st.markdown(f"""
                <div class="result-box">

                <div style="
                font-size:34px;
                font-weight:bold;
                color:black;
                font-family:Times New Roman;
                ">
                🎬 {movie_name}
                </div>

                <br>

                <div style="
                font-size:38px;
                font-weight:bold;
                color:green;
                ">
                😊 Positive Sentiment
                </div>

                <br>

                <div style="
                color:black;
                font-size:30px;
                font-weight:bold;
                ">
                Accuracy: {confidence:.2f}%
                </div>

                </div>
                """, unsafe_allow_html=True)

            # NEGATIVE RESULT
            else:

                st.markdown(f"""
                <div class="result-box">

                <div style="
                font-size:34px;
                font-weight:bold;
                color:black;
                font-family:Times New Roman;
                ">
                🎬 {movie_name}
                </div>

                <br>

                <div style="
                font-size:38px;
                font-weight:bold;
                color:red;
                ">
                😠 Negative Sentiment
                </div>

                <br>

                <div style="
                color:black;
                font-size:30px;
                font-weight:bold;
                ">
                Accuracy: {confidence:.2f}%
                </div>

                </div>
                """, unsafe_allow_html=True)

        else:
            st.warning("⚠️ Please enter a movie review.")

# RIGHT COLUMN
with col2:

    st.markdown("""
    <div class="feature-card">

    <h2 style="
    color:black;
    font-size:34px;
    font-family:Times New Roman;
    font-weight:bold;
    ">
    ✨ Features
    </h2>

    <p style="
    color:black;
    font-size:22px;
    line-height:2;
    font-weight:500;
    ">

    ✅ IMDb Dataset <br>
    ✅ Machine Learning <br>
    ✅ NLP Processing <br>
    ✅ TF-IDF Vectorizer <br>
    ✅ Confidence Score <br>
    ✅ Modern UI Design

    </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-card">

    <h2 style="
    color:black;
    font-size:34px;
    font-family:Times New Roman;
    font-weight:bold;
    ">
    🛠 Tech Stack
    </h2>

    <p style="
    color:black;
    font-size:22px;
    line-height:2;
    font-weight:500;
    ">

    • Python <br>
    • Streamlit <br>
    • Scikit-learn <br>
    • NLP <br>
    • Machine Learning

    </p>

    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
Made with ❤️ using Artificial Intelligence & Machine Learning
</div>
""", unsafe_allow_html=True)