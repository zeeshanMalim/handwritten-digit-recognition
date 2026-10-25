# # webcam_digit_recognition.py

# import numpy as np
# import cv2
# from tensorflow.keras.models import load_model

# # Load model
# model = load_model("model.h5")

# # Start webcam
# cap = cv2.VideoCapture(0)
# print("Press 'q' to quit")

# while True:
#     ret, frame = cap.read()
#     frame = cv2.flip(frame, 1)  # Flip for mirror viewq
#     if not ret:
#         continue

#     # Draw rectangle on screen
#     cv2.rectangle(frame, (100, 100), (300, 300), (255, 0, 0), 2)
#     roi = frame[100:300, 100:300]

#     # Preprocessing
#     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (15, 15), 0)
#     _, thresh = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV)  # Binarize and invert

#     # Resize and reshape
#     resized = cv2.resize(thresh, (28, 28))
#     normalized = resized / 255.0
#     reshaped = normalized.reshape(1, 28, 28)

#     # Predict
#     pred = model.predict(reshaped, verbose=0)
#     digit = np.argmax(pred)
#     confidence = np.max(pred)

#     # Display prediction
#     cv2.putText(frame, f'Prediction: {digit} ({confidence*100:.2f}%)',
#                 (80, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#     # Display window
#     cv2.imshow("Digit Recognition", frame)
#     cv2.imshow("Processed Input", thresh)

#     # Exit on 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# webcam_digit_recognition.py

import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("model.h5")

# Start webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip frame for mirror view
    frame = cv2.flip(frame, 1)

    # Draw ROI rectangle
    x1, y1 = 80, 80
    x2, y2 = 350, 350
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Extract ROI
    roi = frame[y1:y2, x1:x2]

    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Blur image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    # Find contours
    contours, _ = cv2.findContours(
        thresh.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    digit = None
    confidence = 0

    if len(contours) > 0:
        # Largest contour assumed as digit
        c = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(c)

        if w > 10 and h > 10:

            digit_img = thresh[y:y+h, x:x+w]

            # Resize to MNIST size
            digit_img = cv2.resize(digit_img, (28, 28))

            # Normalize
            digit_img = digit_img / 255.0

            # Reshape for model
            digit_img = digit_img.reshape(1, 28, 28, 1)

            # Prediction
            prediction = model.predict(digit_img, verbose=0)

            digit = np.argmax(prediction)
            confidence = np.max(prediction)

    # Display prediction
    if digit is not None:
        cv2.putText(
            frame,
            f"Prediction: {digit} ({confidence*100:.2f}%)",
            (80, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    # Show windows
    cv2.imshow("Digit Recognition", frame)
    cv2.imshow("Threshold", thresh)

    # Quit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()