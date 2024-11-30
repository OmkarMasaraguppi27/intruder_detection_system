import csv
import os
import pickle
import shutil
#import datetime
import time
import tkinter as tk
from datetime import datetime
from tkinter import Message, Text

import cv2
import face_recognition
import numpy as np
import pandas as pd
def predict(rgb_frame, knn_clf=None, model_path=None, distance_threshold=0.5):

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    # X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=2)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    # print(closest_distances)
    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

def identify_faces():

    buf_length = 10
    known_conf = 6
    face_names = []
    buf = [[]] * buf_length
    i = 0

    process_this_frame = True

    # Grab a single frame of video
    vs = cv2.VideoCapture(0)
    
    while True:
       
        ret,image = vs.read()
        (H, W) = image.shape[:2]
        small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = small_frame[:, :, ::-1]
        if process_this_frame:
            predictions = predict(rgb_frame, model_path="./models/trained_model.clf")
        process_this_frame = not process_this_frame
        for name, (top, right, bottom, left) in predictions:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            print("Name==",name)
            face_names.append(name)
            if name=="unknown":
                print("Unknown")
            buf[i] = face_names
            i = (i + 1) % buf_length
            msg=""
            #cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
            #vr.append(LABELS[classIDs[i]])
            #vr.append(x)
            #vr.append(y)

        #print(vr)
        oldvr=vr
        cv2.imshow("Surveillance Camera", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
identify_faces()