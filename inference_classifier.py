import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
load_dotenv()

# ---------------- SET ENVIRONMENT VARIABLE ------------
os.environ['NVIDIA_API_KEY']=os.getenv('NVIDIA_API_KEY')

# ------------------ LOAD MODEL ------------------
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# ------------------ CAMERA ------------------
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

labels_dict = {
    0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',
    10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',
    18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z',
    26:'1',27:'2',28:'3',29:'4',30:'5',31:'6',32:'7',33:'8',
    34:'9',35:' ',36:'OK'
}

# ------------------ STABILITY VARIABLES ------------------
prev_prediction = None
stable_count = 0
last_accept_time = 0

sentence = ""

STABLE_FRAMES = 15
LETTER_DELAY = 1.5  # seconds

# ------------------ LLM FUNCTION ------------------

# Initialize once (outside loop is best)
llm = ChatNVIDIA(
    model="meta/llama-3.3-70b-instruct", 
    temperature=0.4
)

def process_with_llm(sentence: str) -> str:
    """
    Takes recognized sign sentence and generates a meaningful response
    using NVIDIA NIM via LangChain.
    """

    messages = [
        SystemMessage(
            content=(
                "You are a helpful AI assistant. "
                "The user sentence comes from sign language recognition. "
                "Correct minor grammar issues and respond naturally."
            )
        ),
        HumanMessage(content=sentence)
    ]

    response = llm.invoke(messages)

    return response.content


# ------------------ LOOP ------------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        x_, y_, data_aux = [], [], []

        for lm in hand_landmarks.landmark:
            x_.append(lm.x)
            y_.append(lm.y)

        for lm in hand_landmarks.landmark:
            data_aux.append(lm.x - min(x_))
            data_aux.append(lm.y - min(y_))

        prediction = int(model.predict([np.asarray(data_aux)])[0])

        # -------- STABILITY CHECK --------
        if prediction == prev_prediction:
            stable_count += 1
        else:
            stable_count = 0
            prev_prediction = prediction

        current_time = time.time()

        if stable_count >= STABLE_FRAMES and (current_time - last_accept_time) > LETTER_DELAY:
            last_accept_time = current_time
            stable_count = 0

            if prediction == 36:  # OK detected
                print("\n✅ Sentence Completed:", sentence)
                response = process_with_llm(sentence)
                print("🤖 LLM OUTPUT:", response)
                sentence = ""
            else:
                sentence += labels_dict[prediction]
                print("➕ Added:", labels_dict[prediction])

        # -------- UI --------
        char = labels_dict[prediction]
        cv2.putText(frame, f"Pred: {char}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv2.putText(frame, f"Sentence: {sentence}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    cv2.imshow("frame", frame)
    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()
