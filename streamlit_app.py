import streamlit as st
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import os
from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, SystemMessage

# ---------------- CONFIG ----------------
# st.set_page_config(page_title="Sign Language → Sentence", layout="centered")

# Updated page config for wide layout
st.set_page_config(page_title="Sign Language → Sentence", layout="wide")

# ---------------- ENV ----------------
load_dotenv()
os.environ['NVIDIA_API_KEY'] = os.getenv('NVIDIA_API_KEY')

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model_dict = pickle.load(open('./model.p', 'rb'))
    return model_dict['model']

model = load_model()

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# ---------------- LABELS ----------------
labels_dict = {
    0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',
    10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',
    18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z',
    26:'1',27:'2',28:'3',29:'4',30:'5',31:'6',32:'7',33:'8',
    34:'9',35:' ',36:'OK'
}

# ---------------- LLM ----------------
@st.cache_resource
def load_llm():
    return ChatNVIDIA(
        model="meta/llama-3.3-70b-instruct",
        temperature=0.4
    )

llm = load_llm()

def process_with_llm(sentence):
    messages = [
        SystemMessage(
            content=(
                "You are a helpful AI assistant. "
                "The input comes from sign language recognition. "
                "Fix small grammar issues and respond naturally."
            )
        ),
        HumanMessage(content=sentence)
    ]
    return llm.invoke(messages).content

# ---------------- SESSION STATE ----------------
if "run" not in st.session_state:
    st.session_state.run = False

if "sentence" not in st.session_state:
    st.session_state.sentence = ""

if "llm_response" not in st.session_state:
    st.session_state.llm_response = ""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- UI ----------------
# ---------------- UI ----------------
st.title("🤟 Sign Language to Sentence")

col1, col2 = st.columns(2)
with col1:
    if st.button("▶ Start"):
        st.session_state.run = True
with col2:
    if st.button("⏹ Stop"):
        st.session_state.run = False

# Main content: Camera and Response side by side
col_camera, col_response = st.columns(2)

with col_camera:
    st.subheader("📹 Camera Feed")
    frame_window = st.image([])
    sentence_box = st.empty()

with col_response:
    st.subheader("� Your Input & Response")
    
    # Current sentence being built
    sentence_display = st.empty()
    
    # Current LLM response
    response_display = st.empty()



# ---------------- STABILITY ----------------
prev_prediction = None
stable_count = 0
last_accept_time = 0

STABLE_FRAMES = 15
LETTER_DELAY = 1.5

# ---------------- CAMERA LOOP ----------------
if st.session_state.run:
    cap = cv2.VideoCapture(0)

    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not available")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

            x_, y_, data_aux = [], [], []

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            prediction = int(model.predict([np.asarray(data_aux)])[0])
            char = labels_dict[prediction]

            # Stability logic
            if prediction == prev_prediction:
                stable_count += 1
            else:
                stable_count = 0
                prev_prediction = prediction

            current_time = time.time()

            if stable_count >= STABLE_FRAMES and (current_time - last_accept_time) > LETTER_DELAY:
                last_accept_time = current_time
                stable_count = 0

                if prediction == 36:  # OK
                    response = process_with_llm(st.session_state.sentence)
                    st.session_state.llm_response = response
                    st.session_state.chat_history.append({
                        "input": st.session_state.sentence,
                        "output": response
                    })
                    st.session_state.sentence = ""
                else:
                    st.session_state.sentence += char

            cv2.putText(frame, f"Pred: {char}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        sentence_display.markdown(f"**📝 Building:** `{st.session_state.sentence}`")
        if st.session_state.llm_response:
            response_display.markdown(f"**🤖 Response:**\n{st.session_state.llm_response}")
        else:
            response_display.info("Waiting for LLM response...")

    cap.release()
