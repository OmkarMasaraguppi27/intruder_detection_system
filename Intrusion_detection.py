# import libraries
import cv2
import numpy as np
import pandas as pd
import time
import datetime
import smtplib 
from email.message import EmailMessage
import smstest as sms
from playsound import playsound
import cv2
import pickle
import face_recognition
import datetime

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
    print("len(X_face_locations)==",len(X_face_locations))

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
def process():
    buf_length = 10
    known_conf = 5
    buf = [[]] * buf_length
    i = 0

    process_this_frame = True
    
    df=pd.read_csv("PersonDetatils\PersonDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    col_names =  ['Id','Name','Date','Time']
    co=['name']
    attendance = pd.DataFrame(columns = col_names)
    namess=""
    tie = datetime.datetime.now()

    for index, row in df.iterrows():
        namess+= row['Name']+" "
        
    aa=""
    oldaa=""
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("frame_width==",frame_width)
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("frame_height==",frame_height)
    fps = int(cam.get(cv2.CAP_PROP_FPS))
    # Define the codec and create VideoWriter object
    max_contours = 3      # Number of contours to use for rendering a bounding rectangle.
    frame_count = 0
    min_contour_area_thresh = 0.01
    tim = datetime.datetime.now()
    
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))


    
    attendance1 = pd.DataFrame(columns = co) 
    # KNN
    KNN_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows = True) # detectShadows=True : exclude shadow areas from the objects you detected
    # MOG2
    MOG2_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows = True) # exclude shadow areas from the objects you detected
    # choose your subtractor
    bg_subtractor=MOG2_subtractor
    camera = cv2.VideoCapture(0)
    frame=[]
    face=[]
    aa==""
    name=""
    while True:
        ret, frame = camera.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = small_frame[:, :, ::-1]
         #cv2.putText(frame,str(tt),(x,y+h), font, 1,(255,255,255),2)     
        # Every frame is used both for calculating the foreground mask and for updating the background. 
        foreground_mask = bg_subtractor.apply(frame)
        # threshold if it is bigger than 240 pixel is equal to 255 if smaller pixel is equal to 0
        # create binary image , it contains only white and black pixels
        ret , treshold = cv2.threshold(foreground_mask.copy(), 120, 255,cv2.THRESH_BINARY)
        #  dilation expands or thickens regions of interest in an image.
        dilated = cv2.dilate(treshold,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)),iterations = 2)
        # find contours 
        __,contours, hier = cv2.findContours(dilated,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # check every contour if are exceed certain value draw bounding boxes
        if contours:
            # Sort contours based on area.
            recording = True
            contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
            # Contour area of largest contour.
            contour_area_max = cv2.contourArea(contours_sorted[0])
            # Compute fraction of total frame area occupied by largest contour.
            contour_frac = contour_area_max / (frame_width * frame_height)
            # Confirm contour_frac is greater than min_contour_area_thresh threshold.
            if contour_frac > min_contour_area_thresh:
                # Compute bounding rectangle for the top N largest contours.
                for idx in range(min(max_contours, len(contours_sorted))):
                    xc, yc, wc, hc = cv2.boundingRect(contours_sorted[idx])
                    if idx == 0:
                        x1, y1, x2, y2 = xc, yc, xc + wc, yc + hc
                    else:
                        x1, y1, x2, y2 = min(x1, xc), min(y1, yc), max(x2, xc + wc), max(y2, yc + hc)
                # Draw bounding rectangle for top N contours on output frame.
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        
        out.write(frame)
        #cv2.imshow("Subtractor", foreground_mask)
        #cv2.imshow("threshold", treshold)
        if process_this_frame:
            predictions = predict(rgb_frame, model_path="./models/trained_model.clf")
        process_this_frame = not process_this_frame
        for name, (top, right, bottom, left) in predictions:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            print("Name==",name)
        
        ts = time.time()      
        ti = datetime.datetime.now()
        print("ti==",ti)
        face_names=name
        print("face_names==",name)
        if face_names!="unknown":
            aa=df.loc[df['Name'] == face_names]['Name'].values
            mobile=str(df.loc[df['Name'] == face_names]['Email'].values)
            print("Email id==",str(mobile))
            mobile=mobile.replace("[","")
            mobile=mobile.replace("]","")
            mobile=mobile.replace("'","")
            print("Email id==",str(mobile))
            aaa=''.join(e for e in aa if e.isalnum())
            #print aaa
            tt=aa
            
        else:
            cv2.imwrite("out.png",frame)
            Id='Unknown' 
            ti = datetime.datetime.now()
            playsound("alarm.wav")
            print("ti==",ti)      
            tt=str(Id)  
        if tt=='Unknown':
             local = datetime.datetime.now()
             aa1= local.strftime("%M")
             status=0
             if int(aa1)%2==0:
                status=1
                if status==1:
                    msg = EmailMessage()
                    msg.set_content("Mallicious Person entered into your home at "+str(ti)+" : Unsafe ")
                    msg['Subject'] = 'Alert Email'
                    msg['From'] = "otpserver2024@gmail.com"
                    msg['To'] = 'shreyas200379@gmail.com'
                    s = smtplib.SMTP('smtp.gmail.com', 587)
                    s.starttls()
                    s.login("otpserver2024@gmail.com", "hjajxhrufxfmkwta")
                    s.send_message(msg)
                    s.quit()
                    sms.process("Mallicious Person entered into your home at "+str(ti)+" : Unsafe ")
                    
        else:
            local = datetime.datetime.now()
            aa1= local.strftime("%M")
            status=0
            if int(aa1)%5==0:

                status=1
                if status==1:
                    print("aa==",aa)
                    aa=str(aa)
                    aa=aa.replace('[','')
                    aa.replace(']','')
                    aa.replace("'","")
                    print("Final aa==",aa)
                    print("oldaa==",oldaa)
                    if str(oldaa)==str(aa):
                        print("Same Person inside home")
                    else:
                        msg = EmailMessage()
                        print("Predicted==",aa)
                        msgtxt="Mr. "+str(aa)+" Entered in your home at "+str(ti)+" : Safe"
                        print("msgtxt==",msgtxt)
                        msg.set_content(msgtxt)
                        msg['Subject'] = 'Alert Email'
                        msg['From'] = "otpserver2024@gmail.com"
                        msg['To'] = mobile
                        s = smtplib.SMTP('smtp.gmail.com', 587)
                        s.starttls()
                        s.login("otpserver2024@gmail.com", "hjajxhrufxfmkwta")
                        s.send_message(msg)
                        s.quit()
                        oldaa=aa


       
        
        cv2.imshow("detection", frame)
        if cv2.waitKey(30) & 0xff == 27:
            break
    camera.release()
    out.release()
    cv2.destroyAllWindows()
#process()