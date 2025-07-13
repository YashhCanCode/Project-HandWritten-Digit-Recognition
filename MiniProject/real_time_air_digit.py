import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model

# Load your trained model (update the file name if needed)
model = load_model('mnist_cnn.h5')

# MediaPipe Hands init
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Canvas to draw digit
canvas = np.zeros((280, 280), dtype=np.uint8)

def preprocess(img):
    resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    inverted = cv2.bitwise_not(resized)
    _, thresh = cv2.threshold(inverted, 100, 255, cv2.THRESH_BINARY)
    norm = thresh.astype('float32') / 255.0
    return norm.reshape(1, 28, 28, 1)

cap = cv2.VideoCapture(0)  # Open webcam

last_pos = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # Mirror image

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            x = int(hand_landmarks.landmark[8].x * w)  # index finger tip x
            y = int(hand_landmarks.landmark[8].y * h)  # index finger tip y

            if last_pos is not None:
                # Draw on canvas scaled to 280x280
                cv2.line(canvas, (last_pos[0]*280//w, last_pos[1]*280//h), (x*280//w, y*280//h), 255, 15)
            last_pos = (x, y)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        last_pos = None

    cv2.imshow("Webcam Feed", frame)
    cv2.imshow("Canvas", canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # quit
        break
    elif key == ord('c'):  # clear canvas
        canvas.fill(0)
    elif key == ord('p'):  # predict digit
        input_img = preprocess(canvas)
        prediction = model.predict(input_img)
        digit = prediction.argmax()
        print("Predicted digit:", digit)

cap.release()
cv2.destroyAllWindows()
