Real-Time Digit Recognition

This project demonstrates handwritten digit recognition using a deep learning model trained on the MNIST dataset. The trained model is then used to recognize digits drawn in real-time through a webcam.

---

✨ Features

• Train a deep learning model on MNIST dataset
• Save and load the trained model for inference
• Real-time digit recognition using webcam
• ROI-based digit drawing and preprocessing pipeline
• Live prediction with confidence score

---

🛠 Tech Stack

• Python
• TensorFlow / Keras
• OpenCV
• NumPy

---

▶️ Usage

1. Train and save the model

python model_training_and_save.py

2. Run the real-time digit recognition

python webcam_digit_recognition.py


3. Draw digits inside the blue ROI box and view predictions on screen.
   Press Q to quit webcam.

---

📸 Demo

* Draw digits in the blue ROI box (100x100 to 300x300 area)
* The system predicts the digit with confidence
* Example: `Prediction: 7 (95.60%)`