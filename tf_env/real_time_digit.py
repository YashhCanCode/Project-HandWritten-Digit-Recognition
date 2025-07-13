import cv2
import numpy as np
from keras.models import load_model

# Load your trained model (update the path as needed)
model = load_model('mnist_cnn.h5')

# Create a black image for drawing
canvas = np.zeros((280, 280), dtype=np.uint8)

# Window name
window_name = 'Draw a digit (Press q to quit, c to clear)'

def preprocess(img):
    # Resize to 28x28
    resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    # Invert colors (white digit on black bg)
    inverted = cv2.bitwise_not(resized)
    # Threshold to binary
    _, thresh = cv2.threshold(inverted, 100, 255, cv2.THRESH_BINARY)
    # Normalize
    norm = thresh.astype('float32') / 255.0
    # Reshape for model (1, 28, 28, 1)
    return norm.reshape(1, 28, 28, 1)

# Mouse callback function to draw on canvas
drawing = False
last_point = None

def draw(event, x, y, flags, param):
    global drawing, last_point

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        last_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(canvas, last_point, (x, y), (255), thickness=15)
            last_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(canvas, last_point, (x, y), (255), thickness=15)

cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, draw)

while True:
    # Show the canvas (scaled up for visibility)
    display_img = cv2.resize(canvas, (280, 280))
    cv2.imshow(window_name, display_img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):  # Quit
        break
    elif key == ord('c'):  # Clear canvas
        canvas.fill(0)
    elif key == ord('p'):  # Predict on current canvas
        input_img = preprocess(canvas)
        pred = model.predict(input_img)
        digit = pred.argmax()
        print("Predicted digit:", digit)

cv2.destroyAllWindows()
