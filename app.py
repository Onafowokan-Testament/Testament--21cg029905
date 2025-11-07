import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array

# Page configuration
st.set_page_config(
    page_title="Face Emotion Detector",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(120deg, #a1c4fd 0%, #c2e9fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .emotion-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        border-radius: 8px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Constants
MODEL_PATH = "face_emotionModel.h5"
DB_PATH = "emotion_database.db"
UPLOAD_FOLDER = "uploaded_images"
EMOTION_CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Emotion emojis for better visualization
EMOTION_EMOJIS = {
    "Angry": "😠",
    "Disgust": "🤢",
    "Fear": "😨",
    "Happy": "😊",
    "Neutral": "😐",
    "Sad": "😢",
    "Surprise": "😲",
}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Database initialization
def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT,
                  email TEXT,
                  emotion TEXT,
                  confidence REAL,
                  timestamp TEXT,
                  image_path TEXT)"""
    )
    conn.commit()
    conn.close()


# Load model with caching
@st.cache_resource
def load_emotion_model():
    """Load the pre-trained model"""
    try:
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
            return model, None
        else:
            return None, f"Model file not found at {MODEL_PATH}"
    except Exception as e:
        return None, f"Error loading model: {str(e)}"


def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Convert to grayscale and resize
    img = image.convert("L")  # Convert to grayscale
    img = img.resize((48, 48))

    # Convert to array and normalize
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    return img_array


def predict_emotion(model, image):
    """Predict emotion from image"""
    try:
        processed_img = preprocess_image(image)
        predictions = model.predict(processed_img, verbose=0)

        # Get emotion and confidence
        emotion_idx = np.argmax(predictions[0])
        emotion = EMOTION_CLASSES[emotion_idx]
        confidence = float(predictions[0][emotion_idx] * 100)

        # Get all predictions for visualization
        all_predictions = {
            EMOTION_CLASSES[i]: float(predictions[0][i] * 100)
            for i in range(len(EMOTION_CLASSES))
        }

        return emotion, confidence, all_predictions
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None


def save_to_database(name, email, emotion, confidence, image_path):
    """Save prediction to database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute(
            """INSERT INTO predictions 
                     (name, email, emotion, confidence, timestamp, image_path) 
                     VALUES (?, ?, ?, ?, ?, ?)""",
            (name, email, emotion, confidence, timestamp, image_path),
        )
        conn.commit()
        conn.close() 
        return True
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return False



def get_prediction_history(limit=10):
    """Retrieve prediction history from database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            f"SELECT * FROM predictions ORDER BY timestamp DESC LIMIT {limit}", conn
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def get_emotion_statistics():
    """Get emotion statistics for visualization"""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            "SELECT emotion, COUNT(*) as count FROM predictions GROUP BY emotion", conn
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


# Initialize database
init_db()


# Main app
def main():
    # Header
    st.markdown(
        '<h1 class="main-header">🎭 Face Emotion Recognition System</h1>',
        unsafe_allow_html=True,
    )

    # Load model
    model, error = load_emotion_model()

    if error:
        st.error(f"⚠️ {error}")
        st.stop()

    st.success("✅ Model loaded successfully!")

    # Sidebar
    with st.sidebar:
        st.header("📋 Navigation")
        page = st.radio(
            "Choose a page:",
            ["🏠 Home - Detect Emotion", "📊 Statistics", "📜 History"],
        )

        st.markdown("---")
        st.header("ℹ️ About")
        st.info(
            """
        This app uses deep learning to detect emotions from facial expressions.
        
        **Emotions detected:**
        - 😊 Happy
        - 😢 Sad
        - 😠 Angry
        - 😲 Surprise
        - 😨 Fear
        - 🤢 Disgust
        - 😐 Neutral
        """
        )

    # Page routing
    if page == "🏠 Home - Detect Emotion":
        home_page(model)
    elif page == "📊 Statistics":
        statistics_page()
    elif page == "📜 History":
        history_page()


def home_page(model):
    """Main prediction page"""
    st.header("Upload Your Photo")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Your Information")
        name = st.text_input("Full Name", placeholder="Enter your name")
        email = st.text_input("Email", placeholder="your.email@example.com")

        st.subheader("📸 Upload Image")
        upload_method = st.radio(
            "Choose upload method:", ["Upload from device", "Use camera"]
        )

        if upload_method == "Upload from device":
            uploaded_file = st.file_uploader(
                "Choose an image...", type=["jpg", "jpeg", "png"]
            )
        else:
            uploaded_file = st.camera_input("Take a picture")

        predict_button = st.button("🔍 Detect Emotion", type="primary")

    with col2:
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.subheader("📷 Your Image")
            st.image(image, use_container_width=True)

            # Predict on button click
            if predict_button:
                if not name or not email:
                    st.warning("⚠️ Please enter your name and email!")
                else:
                    with st.spinner("🔮 Analyzing your emotion..."):
                        emotion, confidence, all_predictions = predict_emotion(
                            model, image
                        )

                        if emotion:
                            # Save image
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            image_filename = f"{name.replace(' ', '_')}_{timestamp}.jpg"
                            image_path = os.path.join(UPLOAD_FOLDER, image_filename)
                            image.save(image_path)

                            # Save to database
                            save_to_database(
                                name, email, emotion, confidence, image_path
                            )

                            # Display result
                            st.markdown(
                                f"""
                                <div class="emotion-box">
                                    {EMOTION_EMOJIS[emotion]} {emotion}
                                    <br>
                                    <span style="font-size: 1.2rem;">
                                    Confidence: {confidence:.2f}%
                                    </span>
                                </div>
                            """,
                                unsafe_allow_html=True,
                            )

                            st.success(f"✅ Prediction saved successfully, {name}!")

                            # Show all predictions as bar chart
                            st.subheader("� Confidence Scores")
                            df_pred = pd.DataFrame(
                                list(all_predictions.items()),
                                columns=["Emotion", "Confidence"],
                            )
                            df_pred = df_pred.sort_values("Confidence", ascending=True)

                            fig = px.bar(
                                df_pred,
                                x="Confidence",
                                y="Emotion",
                                orientation="h",
                                color="Confidence",
                                color_continuous_scale="Viridis",
                            )
                            fig.update_layout(height=400, showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)


def statistics_page():
    """Statistics and analytics page"""
    st.header("📊 Emotion Detection Statistics")

    df_stats = get_emotion_statistics()

    if df_stats.empty:
        st.info("📭 No predictions yet. Start detecting emotions on the Home page!")
    else:
        col1, col2 = st.columns(2)

        with col1:
            # Pie chart
            st.subheader("Emotion Distribution")
            fig_pie = px.pie(
                df_stats,
                values="count",
                names="emotion",
                title="Distribution of Detected Emotions",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Bar chart
            st.subheader("Emotion Counts")
            fig_bar = px.bar(
                df_stats,
                x="emotion",
                y="count",
                color="count",
                color_continuous_scale="Blues",
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        # Summary metrics
        st.subheader("📈 Summary Metrics")
        total = df_stats["count"].sum()
        most_common = df_stats.loc[df_stats["count"].idxmax(), "emotion"]

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Total Predictions", total)
        metric_col2.metric(
            "Most Common Emotion", f"{EMOTION_EMOJIS[most_common]} {most_common}"
        )
        metric_col3.metric(
            "Unique Users", len(get_prediction_history(limit=1000)["email"].unique())
        )


def history_page():
    """Prediction history page"""
    st.header("📜 Prediction History")

    # Controls
    col1, col2 = st.columns([3, 1])
    with col1:
        limit = st.slider("Number of records to display", 5, 50, 10)
    with col2:
        if st.button("🔄 Refresh"):
            st.rerun()

    df = get_prediction_history(limit=limit)

    if df.empty:
        st.info("📭 No prediction history available yet.")
    else:
        # Display records
        for idx, row in df.iterrows():
            with st.expander(
                f"{row['name']} - {EMOTION_EMOJIS[row['emotion']]} {row['emotion']} ({row['timestamp']})"
            ):
                col1, col2 = st.columns([1, 2])

                with col1:
                    if os.path.exists(row["image_path"]):
                        st.image(row["image_path"], width=200)
                    else:
                        st.warning("Image not found")

                with col2:
                    st.write(f"**Name:** {row['name']}")
                    st.write(f"**Email:** {row['email']}")
                    st.write(
                        f"**Emotion:** {EMOTION_EMOJIS[row['emotion']]} {row['emotion']}"
                    )
                    st.write(f"**Confidence:** {row['confidence']:.2f}%")
                    st.write(f"**Timestamp:** {row['timestamp']}")


if __name__ == "__main__":
    main()
