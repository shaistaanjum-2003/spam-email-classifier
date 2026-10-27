import streamlit as st
import joblib
import os
import re
import math

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"] {
    background-color: #f1f4ff;
}

.sidebar-title {
    font-size: 22px;
    font-weight: 700;
    color: #172554;
}

.sidebar-subtitle {
    font-size: 13px;
    color: #64748b;
}

.dashboard-title {
    font-size: 38px;
    font-weight: 800;
    color: #172554;
    margin-bottom: 0px;
}

.dashboard-subtitle {
    color: #64748b;
    font-size: 16px;
}

.section-title {
    font-size: 20px;
    font-weight: 700;
    color: #172554;
}

.ham-box {
    background: linear-gradient(90deg, #e9fff1, #f5fff8);
    border-radius: 15px;
    padding: 20px;
    border-left: 6px solid #16a34a;
}

.spam-box {
    background: linear-gradient(90deg, #fff0f0, #fff7f7);
    border-radius: 15px;
    padding: 20px;
    border-left: 6px solid #dc2626;
}

.ham-title {
    color: #15803d;
    font-size: 25px;
    font-weight: 800;
}

.spam-title {
    color: #dc2626;
    font-size: 25px;
    font-weight: 800;
}

.info-card {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    min-height: 120px;
}

.info-title {
    font-weight: 700;
    color: #334155;
}

.info-text {
    color: #64748b;
    font-size: 14px;
}

.metric-number {
    font-size: 25px;
    font-weight: 800;
    color: #1e3a8a;
}

button[kind="primary"] {
    border-radius: 10px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL AND VECTORIZER
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load("models/spam_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

    return model, vectorizer


try:
    model, vectorizer = load_model()
    model_loaded = True

except Exception as e:
    model_loaded = False
    model = None
    vectorizer = None

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🛡️ Spam Email<br>Classifier</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">Machine Learning Email Detection System</div>',
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "ℹ️ About Project",
            "🤖 Model Information",
            "💡 How it Works",
            "📧 Examples",
            "📞 Contact"
        ]
    )

    st.divider()

    st.markdown("### 🛠 Technologies Used")

    st.markdown("""
    - 🐍 Python
    - 🤖 Scikit-learn
    - 📊 TF-IDF Vectorizer
    - 🎨 Streamlit
    - 💾 Joblib
    - 🔗 GitHub
    """)

# =========================================================
# HOME PAGE
# =========================================================

if page == "🏠 Home":

    # Banner
    banner_path = "banner.png"

    if os.path.exists(banner_path):
        st.image(banner_path, use_container_width=True)

    else:
        st.markdown(
            """
            <div style="
                background: linear-gradient(90deg,#dbeafe,#eef2ff);
                padding:30px;
                border-radius:15px;
                text-align:center;
            ">
                <div class="dashboard-title">
                    🛡️ Spam Email Classifier
                </div>
                <div class="dashboard-subtitle">
                    Detect whether an email is Spam or Ham using Machine Learning
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    st.markdown(
        '<div class="section-title">📝 Enter Email</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Paste or type the email content below and click **Predict Email**."
    )

    email_text = st.text_area(
        "",
        height=160,
        placeholder="Enter or paste your email here..."
    )

    predict_button = st.button(
        "🔍 Predict Email",
        type="primary",
        use_container_width=True
    )

    if predict_button:

        if not model_loaded:

            st.error(
                "Model files could not be loaded. "
                "Please make sure spam_model.pkl and "
                "tfidf_vectorizer.pkl are inside the models folder."
            )

        elif not email_text.strip():

            st.warning("Please enter an email before making a prediction.")

        else:

            # =================================================
            # PREDICTION
            # =================================================

            email_vector = vectorizer.transform([email_text])

            prediction = model.predict(email_vector)[0]

            # Model score
            score = None

            if hasattr(model, "decision_function"):

                decision = model.decision_function(email_vector)[0]

                # Convert decision score into an easy-to-read
                # approximate score. This is NOT a probability.
                score = 1 / (1 + math.exp(-abs(float(decision))))

                score = score * 100

            # =================================================
            # PREDICTION RESULT
            # =================================================

            if str(prediction).lower() == "spam":

                st.markdown(
                    f"""
                    <div class="spam-box">
                        <div class="spam-title">
                            🚨 SPAM EMAIL
                        </div>
                        <p>
                            This email is classified as spam.
                            It may contain unwanted, promotional,
                            suspicious, or potentially harmful content.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="ham-box">
                        <div class="ham-title">
                            ✅ HAM EMAIL
                        </div>
                        <p>
                            This email is classified as Ham (Legitimate).
                            It appears to be a normal and genuine email.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.write("")

            # =================================================
            # MODEL SCORE
            # =================================================

            if score is not None:

                col1, col2, col3 = st.columns([2, 2, 2])

                with col2:

                    st.metric(
                        "Model Score",
                        f"{score:.1f}%"
                    )

            # =================================================
            # EMAIL STATISTICS
            # =================================================

            st.markdown(
                '<div class="section-title">📊 Email Statistics</div>',
                unsafe_allow_html=True
            )

            words = len(email_text.split())
            characters = len(email_text)

            reading_time = max(
                0.1,
                round(words / 150, 1)
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                st.markdown(
                    f"""
                    <div class="info-card">
                        <div class="info-title">
                            📝 Number of Words
                        </div>
                        <div class="metric-number">
                            {words}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with c2:

                st.markdown(
                    f"""
                    <div class="info-card">
                        <div class="info-title">
                            🔤 Number of Characters
                        </div>
                        <div class="metric-number">
                            {characters}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with c3:

                st.markdown(
                    f"""
                    <div class="info-card">
                        <div class="info-title">
                            ⏱️ Estimated Reading Time
                        </div>
                        <div class="metric-number">
                            {reading_time} min
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.write("")

            # =================================================
            # EXPLANATION
            # =================================================

            left, right = st.columns(2)

            with left:

                st.markdown(
                    """
                    <div class="info-card">
                        <h3>💡 Why this prediction?</h3>
                        <p class="info-text">
                        The email text is converted into numerical
                        features using TF-IDF. The trained SVC
                        machine-learning model then uses these
                        features to classify the email as Spam or Ham.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with right:

                st.markdown(
                    """
                    <div class="info-card">
                        <h3>⚠️ Common Spam Indicators</h3>
                        <p class="info-text">
                        • Suspicious links or attachments<br>
                        • Urgent or threatening language<br>
                        • Unexpected prizes or offers<br>
                        • Requests for personal information<br>
                        • Promotional or irrelevant content
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# =========================================================
# ABOUT PROJECT
# =========================================================

elif page == "ℹ️ About Project":

    st.title("ℹ️ About This Project")

    st.write("""
    The Spam Email Classifier is a Machine Learning project
    designed to classify emails into two categories:

    **Spam** – unwanted or suspicious email.

    **Ham** – legitimate or normal email.

    The project uses TF-IDF to convert email text into numerical
    features and an SVC-based machine learning classifier to
    perform the final classification.
    """)

# =========================================================
# MODEL INFORMATION
# =========================================================

elif page == "🤖 Model Information":

    st.title("🤖 Model Information")

    st.markdown("""
    ### Machine Learning Model

    **Algorithm:** Support Vector Classifier (SVC)

    ### Feature Extraction

    **TF-IDF Vectorizer**

    TF-IDF converts textual email data into numerical features
    that can be processed by the machine learning model.

    ### Model Files

    - `spam_model.pkl`
    - `tfidf_vectorizer.pkl`

    These files are loaded using Joblib.
    """)

# =========================================================
# HOW IT WORKS
# =========================================================

elif page == "💡 How it Works":

    st.title("💡 How the Classifier Works")

    st.markdown("""
    ### Step 1 — Email Input

    The user enters an email into the application.

    ### Step 2 — TF-IDF

    The email text is converted into numerical features using
    the trained TF-IDF vectorizer.

    ### Step 3 — SVC

    The numerical features are passed to the trained SVC model.

    ### Step 4 — Classification

    The model predicts:

    - 🟢 Ham
    - 🔴 Spam

    ### Step 5 — Result

    The prediction is displayed on the Streamlit dashboard.
    """)

# =========================================================
# EXAMPLES
# =========================================================

elif page == "📧 Examples":

    st.title("📧 Email Examples")

    st.subheader("🟢 Ham Example")

    st.code("""
Hello,

The meeting has been scheduled for tomorrow at 10 AM.
Please bring the required documents.

Regards,
Team
""")

    st.subheader("🔴 Spam Example")

    st.code("""
Congratulations!

You have won a $10,000 cash prize.
Click the link below immediately to claim your reward.
""")

# =========================================================
# CONTACT
# =========================================================

elif page == "📞 Contact":

    st.title("📞 Contact")

    st.write("""
    **Spam Email Classifier**

    Machine Learning Internship Project

    Developed using Python, Scikit-learn, TF-IDF and Streamlit.
    """)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Spam Email Classifier | Machine Learning Project"
)