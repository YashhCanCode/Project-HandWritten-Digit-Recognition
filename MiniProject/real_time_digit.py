import cv2
import numpy as np
from keras.models import load_model

# Load your trained MNIST model
model = load_model('mnist_cnn.h5')

# Create a black image canvas
canvas = np.zeros((280, 280), dtype=np.uint8)

# Mouse callback function to draw on canvas
drawing = False

def draw(event, x, y, flags, param):
    global drawing, canvas
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(canvas, (x, y), 8, (255), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

cv2.namedWindow('Draw a digit')
cv2.setMouseCallback('Draw a digit', draw)

while True:
    cv2.imshow('Draw a digit', canvas)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('p'):  # Predict when 'p' pressed
        # Prepare the image for prediction
        img = cv2.resize(canvas, (28, 28))
        img = img.astype('float32') / 255.0
        img = img.reshape(1, 28, 28, 1)
        
        prediction = model.predict(img)
        digit = np.argmax(prediction)
        print(f'Predicted Digit: {digit}')
        
    elif key == ord('c'):  # Clear canvas when 'c' pressed
        canvas[:] = 0
        
    elif key == ord('q'):  # Quit when 'q' pressed
        break

cv2.destroyAllWindows()
