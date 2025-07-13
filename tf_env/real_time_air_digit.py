import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model("mnist_cnn_model.h5")

cap = cv2.VideoCapture(0)

canvas = np.zeros((480, 640), dtype=np.uint8)
is_drawing = False
prev_x, prev_y = None, None

print("Press 'd' to start drawing, 'p' to predict, 'c' to clear, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Simulated finger position: center of screen (replace with actual tracking if needed)
    x, y = 320, 240

    if is_drawing:
        if prev_x is not None and prev_y is not None:
            cv2.line(canvas, (prev_x, prev_y), (x, y), 255, 12)
        prev_x, prev_y = x, y
    else:
        prev_x, prev_y = None, None

    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    combined = cv2.addWeighted(frame, 0.5, canvas_bgr, 0.5, 0)
    cv2.imshow("Air Digit Drawer", combined)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('d'):
        is_drawing = True
        print("Drawing mode ON")

    elif key == ord('c'):
        canvas = np.zeros((480, 640), dtype=np.uint8)
        print("Canvas cleared")
        is_drawing = False

    elif key == ord('p'):
        x, y, w, h = cv2.boundingRect(canvas)
        if w > 0 and h > 0:
            digit = canvas[y:y+h, x:x+w]

            padded_digit = cv2.copyMakeBorder(digit, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0)
            roi = cv2.resize(padded_digit, (28, 28))
            roi = cv2.GaussianBlur(roi, (3, 3), 0)
            _, roi = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY)
            roi = roi.astype('float32') / 255.0
            roi = roi.reshape(1, 28, 28, 1)

            pred = model.predict(roi)
            predicted_digit = np.argmax(pred)
            confidence = np.max(pred)

            print(f"Predicted Digit: {predicted_digit} (Confidence: {confidence:.2f})")
        else:
            print("Nothing drawn to predict.")

    elif key == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
