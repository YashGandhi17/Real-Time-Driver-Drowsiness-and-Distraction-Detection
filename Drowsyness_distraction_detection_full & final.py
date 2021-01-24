#####All functions#####
#Drowsyness
#Distraction
#Alarm
#Face detected or not
#ROI
#Image Enhancement
#######################q
from scipy.spatial import distance as dist
import dlib
import cv2
import playsound
from threading import Thread


shape_preditor_path="shape_predictor_68_face_landmarks.dat"
alarm_sound_path="alarm.wav"

i=input()
if (i=="0"):
    i=0
cap=cv2.VideoCapture(i)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_preditor_path)

#face_cascade = cv2.CascadeClassifier("cascades/haarcascade_fontalface_default.xml")
#eye_cascade = cv2.CascadeClassifier("cascades/haarcascade_eye.xml")

def sound_alarm(path):
    playsound.playsound(path)

def detect_faces(img):
    faces=detector(img)
    #faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
    return faces

def plot_landmarks(start,end,img,landmarks):

     for i in range(start,end+1):
       x = landmarks.part(i).x
       y = landmarks.part(i).y
       cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

def select_ROI(faces):
    area=8000
    return_index=0
    index=0
    flag=0
    for face in faces:
        a=(face.right()-face.left())*(face.bottom()-face.top())
        if a>area:
            area=a
            flag=1
            return_index=index
        index+=1
    return return_index,flag

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[3],mouth[9])
    B = dist.euclidean(mouth[2], mouth[10])
    C = dist.euclidean(mouth[4], mouth[8])
    D = dist.euclidean(mouth[0],mouth[6])

    mar= (A+B+C)/(3.0*D)
    return mar

def side_aspect_ratio(face,nose):
    diff=0
    for i in range(9):
        A = dist.euclidean(face[i], nose)
        B = dist.euclidean(face[16-i], nose)
        diff+=abs(A-B)
    return diff

def head_aspect_ratio(nose):
    A = dist.euclidean(nose[0], nose[1])
    return A

def get_coordinate(p, landmarks):
    return [(landmarks.part(p).x, landmarks.part(p).y)]

def get_coordinates(start, end, landmarks):
    coordinates=[]
    for i in range(start, end+1):
        coordinates.append((landmarks.part(i).x, landmarks.part(i).y))
    return coordinates

def is_face_detected(face):
    detected=True
    if not face:
        detected=False
    return detected

#variables
EYE_AR_THRESH=0.3
EYE_CONSEQ_FRAME1=45
EYE_CONSEQ_FRAME2=60

MOUTH_AR_THRESH=0.7
MOUTH_CONSEQ_FRAME1=45  #1 second
MOUTH_CONSEQ_FRAME2=60  #1.5 second

MOE_CONSEQ_FRAME1=45
MOE_THRESH=1.75

SIDE_AR_THRESH=200
SIDE_CONSEQ_FRAME1=60

HEAD_AR_THRESH1=20
HEAD_AR_THRESH2=38
HEAD_CONSEQ_FRAME1=60

FACE_CONSEQ_FRAME1=60
#Counter
E_COUNTER=0
M_COUNTER=0
MOE_COUNTER=0
SAR_COUNTER=0
HEAD_COUNTER=0
FACE_COUNTER=0
ALARM1_ON=False

c=0
while cap.isOpened():
    ret, frame = cap.read()
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE()
    enc_img = clahe.apply(gray_img)
    faces = detect_faces(enc_img)
    c+=1
    print(c)
    face_detected = is_face_detected(faces)
    if not face_detected:
        c = 0
        FACE_COUNTER += 1  # +1 if face not detected
        # print("No face detected")
        if FACE_COUNTER >= FACE_CONSEQ_FRAME1:
            if not ALARM1_ON:
                ALARM1_ON = True
                t = Thread(target=sound_alarm, args=(alarm_sound_path,))
                t.start()
                FACE_COUNTER = 0
                print("ALARM1_ON by FACE not detected.")

                cv2.putText(frame, "Face not Detected", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 3)
                ALARM1_ON = False
    else:
        FACE_COUNTER = 0

    if len(faces)>0:
        face_index,flag = select_ROI(faces)
        if flag==1:
            face = faces[face_index]
            landmarks = predictor(gray_img, face)
            plot_landmarks(0, 35, frame, landmarks)



        left_eye = get_coordinates(42, 47, landmarks)
        right_eye = get_coordinates(36, 41, landmarks)
        mouth = get_coordinates(48, 67, landmarks)
        face = get_coordinates(0, 16, landmarks)
        nose = get_coordinates(33, 33, landmarks)
        points = []
        points +=  get_coordinate(27, landmarks) + get_coordinate(30, landmarks)

        leftEAR=eye_aspect_ratio(left_eye)
        rightEAR = eye_aspect_ratio(right_eye)
        MAR = mouth_aspect_ratio(mouth)
        EAR=(leftEAR+rightEAR)/2.0
        MOE=MAR/EAR
        SAR = side_aspect_ratio(face, nose[0])
        HAR = head_aspect_ratio(points)
        if c == 1:
            HEAD_AR_THRESH1 = HAR - 7
            HEAD_AR_THRESH2 = HAR + 5
            print(HAR)  # For debug purpose

        #MOE Checking
        if MOE>MOE_THRESH:
            MOE_COUNTER+=1

            if MOE_COUNTER>=MOE_CONSEQ_FRAME1:
                if not ALARM1_ON:
                    ALARM1_ON=True
                    t = Thread(target=sound_alarm, args=(alarm_sound_path,))
                    t.start()
                    MOE_COUNTER=0
                    print("ALARM1_ON by MOE")

                    cv2.putText(frame,"Drowsiness Detected",(15,15),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),3)
                    ALARM1_ON = False
        else:
           MOE_COUNTER = 0

        #SAR Checking
        if SAR>SIDE_AR_THRESH:
            SAR_COUNTER+=1

            if SAR_COUNTER>=SIDE_CONSEQ_FRAME1:
                if not ALARM1_ON:
                    ALARM1_ON=True
                    t = Thread(target=sound_alarm, args=(alarm_sound_path,))
                    t.start()
                    SAR_COUNTER=0
                    print("ALARM1_ON by SAR")

                    cv2.putText(frame,"Drowsiness Detected",(15,15),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),3)
                    ALARM1_ON = False
        else:
           SAR_COUNTER = 0

        #HAR Checking
        if HAR < HEAD_AR_THRESH1 or HAR > HEAD_AR_THRESH2:
            HEAD_COUNTER += 1

            if HEAD_COUNTER >= HEAD_CONSEQ_FRAME1:
                if not ALARM1_ON:
                    ALARM1_ON = True
                    t = Thread(target=sound_alarm, args=(alarm_sound_path,))
                    t.start()
                    HEAD_COUNTER = 0
                    print("ALARM1_ON by UP_DOWN")

                    cv2.putText(frame, "Drowsiness Detected", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 3)
                    ALARM1_ON = False
        else:
            HEAD_COUNTER = 0


        cv2.putText(frame, "EAR:{:.2f}".format(EAR),(300,30),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),2)
        cv2.putText(frame, "MAR:{:.2f}".format(MAR), (300, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)
        cv2.putText(frame, "SAR:{:.2f}".format(SAR),(300,70),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),2)
        cv2.putText(frame, "HAR:{:.2f}".format(HAR),(300,90),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),2)

    cv2.imshow('f',frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break
cv2.destroyAllWindows()
cap.release()

