import cv2 as cv
import numpy as np
import os

print("OpenCV version:", cv.__version__)

def compare_faces(face1, face2):
    """Compare two face images using template matching"""

    if len(face1.shape) == 3:
        gray1 = cv.cvtColor(face1, cv.COLOR_BGR2GRAY)
    else:
        gray1 = face1

    if len(face2.shape) == 3:
        gray2 = cv.cvtColor(face2, cv.COLOR_BGR2GRAY)
    else:
        gray2 = face2


    h1, w1 = gray1.shape
    h2, w2 = gray2.shape

    if h1 * w1 < h2 * w2:

        gray1 = cv.resize(gray1, (w2, h2))
    elif h2 * w2 < h1 * w1:

        gray2 = cv.resize(gray2, (w1, h1))

    # template matching
    try:
        result = cv.matchTemplate(gray1, gray2, cv.TM_CCOEFF_NORMED)
        similarity = result[0][0]

        return 1.0 - similarity  
    except:
        return 1.0  
def load_known_faces(reference_images):
    """Load multiple reference faces"""
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    known_faces = {}

    for image_path, person_name in reference_images.items():
        if not os.path.exists(image_path):
            print(f"Warning: {image_path} not found, skipping {person_name}")
            continue

        reference_img = cv.imread(image_path)
        if reference_img is None:
            print(f"Error: Could not load {image_path}")
            continue

        reference_gray = cv.cvtColor(reference_img, cv.COLOR_BGR2GRAY)
        reference_faces = face_cascade.detectMultiScale(reference_gray, 1.1, 5)

        if len(reference_faces) == 0:
            print(f"Warning: No face found in {image_path}")
            continue

        # Extract face from reference
        (x, y, w, h) = reference_faces[0]

        padding_x = int(w * 0.1)
        padding_y = int(h * 0.1)
        x = max(0, x - padding_x)
        y = max(0, y - padding_y)
        w = min(reference_img.shape[1] - x, w + 2 * padding_x)
        h = min(reference_img.shape[0] - y, h + 2 * padding_y)
        known_faces[person_name] = reference_img[y:y+h, x:x+w]
        print(f"Reference face learned for {person_name}")

    return known_faces

def recognize_multiple_faces(known_faces):
    """Detect and recognize multiple faces in camera feed"""
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

    if not known_faces:
        print("Error: No known faces loaded")
        return

    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:

            padding_x = int(w * 0.1)
            padding_y = int(h * 0.1)
            x_pad = max(0, x - padding_x)
            y_pad = max(0, y - padding_y)
            w_pad = min(frame.shape[1] - x_pad, w + 2 * padding_x)
            h_pad = min(frame.shape[0] - y_pad, h + 2 * padding_y)
            detected_face = frame[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]


            best_match = None
            best_similarity = float('inf')
            best_name = "Unknown"

            for name, known_face in known_faces.items():
                similarity = compare_faces(known_face, detected_face)
                print(f"Comparing with {name}: similarity = {similarity:.3f}")  # Debug output
                if similarity < best_similarity:
                    best_similarity = similarity
                    best_name = name

            print(f"Best match: {best_name} with similarity {best_similarity:.3f}")

            # threshhold
            if best_similarity < 0.4:  
                label = best_name
                color = (0, 255, 0)  
            else:
                label = "Unknown"
                color = (0, 0, 255)  

            cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            cv.putText(frame, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX,
                       1, color, 2)

        cv.imshow('Multi-Face Recognition', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    reference_images = {
        "test1.jpg": "test1",
        "test2.jpg": "test2",
        "test3.jpg": "test3"
    }


    known_faces = load_known_faces(reference_images)

    if known_faces:
        print(f"Loaded {len(known_faces)} known faces: {list(known_faces.keys())}")
        recognize_multiple_faces(known_faces)
    else:
        print("Error: No reference faces could be loaded. Please ensure you have mohamed.jpg and midou.jpg in the project folder")
