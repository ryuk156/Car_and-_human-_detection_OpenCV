
import cv2

img_file="car2.jpg"
video=cv2.VideoCapture('motor.mp4')


#trained car data
classifier_car='car_dect.xml'
classifier_human='haarcascade_fullbody.xml'


#runvideo
while True:
    read_sucessful,frame=video.read()
    
    #safecode
    if read_sucessful:
        gray_video=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    else:
        break
    car_tracker=cv2.CascadeClassifier(classifier_car)
    human_tracker=cv2.CascadeClassifier(classifier_human)
    car=car_tracker.detectMultiScale(gray_video)
    human=human_tracker.detectMultiScale(gray_video)
    for (x,y,w,h) in car:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    for (x,y,w,h) in human:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow('car image',frame)

    cv2.waitKey(1)


"""

#create opencv image
img=cv2.imread(img_file)

#convert image black and white i.e. grayscale
black_and_white=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#create classifier
car_tracker=cv2.CascadeClassifier(classifier_file)

#detect cars
cars=car_tracker.detectMultiScale(black_and_white)

for (x,y,w,h) in cars:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)





#display image
cv2.imshow('car image',img)

cv2.waitKey()

print('complete')

"""