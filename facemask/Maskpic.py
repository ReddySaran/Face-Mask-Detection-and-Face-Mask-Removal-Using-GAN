import cv2
import imutils
from imutils.video import VideoStream
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import os
import face_recognition
from django.core.files.uploadedfile import InMemoryUploadedFile
import pyttsx3

prototxtPath = "C:/Users/USER/OneDrive/Desktop/Django/facemask/deploy.prototxt.txt"
weightsPath = "C:/Users/USER/OneDrive/Desktop/Django/facemask/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model("C:/Users/USER/OneDrive/Desktop/Django/facemask/mask_detector.model")

IMAGE_FOLDER_PATH = "C:/Users/USER/OneDrive/Desktop/face mask/images"

person_image_paths = []
person_names = []

# List files in the directory
for filename in os.listdir(IMAGE_FOLDER_PATH):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        person_image_paths.append(os.path.join(IMAGE_FOLDER_PATH, filename))

        name = os.path.splitext(filename)[0]
        name = name.title()
        person_names.append(name)

# Load known face encodings and names from image files
def load_known_faces(image_paths, names):
    known_face_encodings = []
    known_names = []
    for image_path, name in zip(image_paths, names):
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        known_face_encodings.extend(face_encodings)
        known_names.extend([name] * len(face_encodings))

    return known_face_encodings, known_names

# Load known faces
known_face_encodings, known_names = load_known_faces(person_image_paths, person_names)

class maskdetect(object):
    def __init__(self):
        self.vs=VideoStream(src=0).start()
    
        
    def __del__(self):
        cv2.destroyAllWindows()
        self.vs.stop()

    def detect_and_predict_mask(self,frame, faceNet, maskNet):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
        faceNet.setInput(blob)
        detections = faceNet.forward()
        faces = []
        locs = []
        preds = []

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                faces.append(face)
                locs.append((startX, startY, endX, endY))

        if len(faces) > 0:
            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=32)

        return (locs, preds)

    # Function to handle face mask detection
    def mask_detection(self,frame, faceNet, maskNet):
        (locs, preds) = self.detect_and_predict_mask(frame, faceNet, maskNet)

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    def recognize_faces(self,frame, known_face_encodings, known_names):
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            top, right, bottom, left = face_location
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            print("Recognized name:", name)
            # voice=name +"wear the mask"
            # engine = pyttsx3.init()
            # voices = engine.getProperty('voices')
            # engine.setProperty('voice', voices[1].id)
            # engine.say(voice)
            # engine.runAndWait()
    def process_frame(self,frame, faceNet, maskNet, known_face_encodings, known_names):
        self.recognize_faces(frame, known_face_encodings, known_names)
        self.mask_detection(frame, faceNet, maskNet)

    def get_frame(self):

        frame=self.vs.read()
        frame=imutils.resize(frame,width=700)
        frame=cv2.flip(frame,1)
        self.process_frame(frame, faceNet, maskNet, known_face_encodings, known_names)
        # self.recognize_faces(frame,known_face_encodings, known_names)
        # self.mask_detection(frame, faceNet, maskNet)
        ret,jpeg=cv2.imencode('.jpg',frame)
        return jpeg.tobytes()
    


class maskdetect1(object):
    def detect_and_predict_mask(self,frame, faceNet, maskNet):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
        faceNet.setInput(blob)
        detections = faceNet.forward()
        faces = []
        locs = []
        preds = []

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                faces.append(face)
                locs.append((startX, startY, endX, endY))

        if len(faces) > 0:
            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=32)

        return (locs, preds)

    # Function to handle face mask detection
    def mask_detection(self,frame, faceNet, maskNet):
        (locs, preds) = self.detect_and_predict_mask(frame, faceNet, maskNet)

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    def recognize_faces(self,frame, known_face_encodings, known_names):
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            top, right, bottom, left = face_location
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            print("Recognized name:", name)

    def resize_image(self,image_file, width=400):
        if isinstance(image_file, InMemoryUploadedFile):
            nparr = np.frombuffer(image_file.read(), np.uint8)
            image = cv2.imdecode(nparr, cv2.COLOR_RGB2GRAY)
        else:
            return None
        resized_image = imutils.resize(image, width=width)

        return resized_image
    
    def get_frame1(self,input_image):
        if input_image is None:
            print("Error: Frame is None")
            return None
        frame1=self.resize_image(input_image)
        self.recognize_faces(frame1,known_face_encodings, known_names)
        self.mask_detection(frame1, faceNet, maskNet)
        ret,jpeg=cv2.imencode('.jpg',frame1)
        return jpeg.tobytes()


