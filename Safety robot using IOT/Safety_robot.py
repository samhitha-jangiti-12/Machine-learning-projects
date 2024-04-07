from PIL import Image
import base64
import Adafruit_CharLCD as LCD
import RPi.GPIO as GPIO
import time
from datetime import datetime
import os
from smbus import SMBus
import math
import cv2
import time
import shutil
import sys
from glob import glob
from subprocess import check_output, CalledProcessError
import requests

bus = SMBus(1)
url = 'http://iotgecko.com/IOTImgAdd.aspx?id=samhitha.jangiti@gmail.com&pass=8250'


lcd_rs        = 21  # Note this might need to be changed to 21 for older revision Pi's.
lcd_en        = 20
lcd_d4        = 16
lcd_d5        = 12
lcd_d6        = 7
lcd_d7        = 8

lcd_columns = 16
lcd_rows    = 2

lcd = LCD.Adafruit_CharLCD(lcd_rs, lcd_en, lcd_d4, lcd_d5, lcd_d6, lcd_d7,
                           lcd_columns, lcd_rows)

back_trigg = 15
back_echo = 14
front_trigg= 9
front_echo = 5

button = 4

m10 = 13
m11 = 6
m20 = 26
m21 = 19

f_ir1 = 18
f_ir2 = 27
b_ir1 = 22
b_ir2 = 23

GPIO.setup(back_trigg, GPIO.OUT)
GPIO.setup(back_echo, GPIO.IN)
GPIO.setup(front_trigg, GPIO.OUT)
GPIO.setup(front_echo, GPIO.IN)

GPIO.setup(button, GPIO.IN)
GPIO.setup(m11,GPIO.OUT)
GPIO.setup(m10,GPIO.OUT)
GPIO.setup(m21,GPIO.OUT)
GPIO.setup(m20,GPIO.OUT)
GPIO.setup(f_ir1,GPIO.IN)
GPIO.setup(f_ir2,GPIO.IN)
GPIO.setup(b_ir1,GPIO.IN)
GPIO.setup(b_ir2,GPIO.IN)

GPIO.output(m11, False)## stop
GPIO.output(m10, False)
GPIO.output(m21, False)
GPIO.output(m20, False)

direction = ['backward' , 'forward']
max_val = 150

def check_connectivity():
    try:
        ret, frame = camera1.read()
        print ('sending sample image')
        t1 = datetime.now()
        for x in range(10):
            ret, frame = camera1.read()
            cv2.imwrite("pic2.jpeg", frame)
        with open('pic2.jpeg', 'rb') as f:
            en = base64.b64encode(f.read())
        data ={'img':en}
        r = requests.post(url, data= data)
        res = r.text
        print(res)
        print(type(res))
        if res.find("True") > 0 or res.find("true") > 0:
            print('true')
            return True
        elif res.find("Error") > 0 or res.find("error") > 0: 
            print('flase')
            return False
    except:
        return False

def send_iot_data():
    try:
        print ('sending sample image')
        with open('pic2.jpeg', 'rb') as f:
            en = base64.b64encode(f.read())
        data = {'img':en}
        r = requests.post(url, data= data)
        res = r.text
        print(res)
        if res.find("True"):
            return True
        elif res.find("Error"):
            return False
    except:
        return False

def distance_isr(Trigger, Echo):
    global distance
    GPIO.output(Trigger, False)
    time.sleep(0.60)
    GPIO.output(Trigger, True)
    time.sleep(0.00001)
    GPIO.output(Trigger, False)
 
    StopTime = time.time()
    while GPIO.input(Echo) == 0:
        None
  
    StartTime = time.time()
    while GPIO.input(Echo) == 1:
        None

    TimeElapsed = time.time() - StartTime
    distance = (TimeElapsed * 34300) / 2

    if distance >= 200:
        distance = 200
    return distance

def adc(chn):
    if chn == 0:
        bus.write_byte(0x48,0x40)
    if chn == 1:
        bus.write_byte(0x48,0x41)
    bus.read_byte(0x48)
    return int(bus.read_byte(0x48))

def check_sound():
    direct = "None"
    r = adc(0)
    f = adc(1)
    r = r 
    list1 = [r,f]
    print (list1)
    if r >= 120 or f >= max_val :
        print (r , f )
        i = max(list1)
        for s in range(len(list1)):
            if list1[s] == i:
                print (direction[s] , i)
                direct = direction[s]
                break
    return direct

def detect_face(camera):
    
    face_cascade = cv2.CascadeClassifier('/home/pi/haarcascade_frontalface_default.xml')
    
    cam = camera
    
    face = False
    for x in range(10):
        ret, frame = cam.read()
##        cv2.imshow('img',frame)
##        cv2.waitKey(1)
    
    cv2.imwrite("pic2.jpeg", frame)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
##    hog = cv2.HOGDescriptor() ## for human detection
##    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
##    (faces, weights) = hog.detectMultiScale(gray, winStride=(4, 4),
##		padding=(8, 8), scale=1.05)
    for (x,y,w,h) in faces:
        print ("face detected")
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        face = True
##    cv2.imshow('gray',gray)
##    cv2.imshow('img',frame)
    
    if face:
        return True
    else:
        return False
    
def drive_motor(direct):
    global isr_direct
    isr_direct = direct
    print('dir: ', direct)
    if direct == 'backward':
        GPIO.output(m11, False)## back
        GPIO.output(m10, True)
        GPIO.output(m21, True)
        GPIO.output(m20, False)
        
    if direct == 'forward':
        GPIO.output(m11, True)## forward
        GPIO.output(m10, False)
        GPIO.output(m21, False)
        GPIO.output(m20, True)

    if direct == 'stop':
        GPIO.output(m11, False)## stop
        GPIO.output(m10, False)
        GPIO.output(m21, False)
        GPIO.output(m20, False)

    
def f_ir1_detect(channel1):##left sensor
    print('fir1')
    GPIO.output(m11, False)#rurn right
    GPIO.output(m10, False)
    GPIO.output(m21, False)
    GPIO.output(m20, True)
    time.sleep(0.2)
    drive_motor('forward')
        
def f_ir2_detect(channel2):##right sensor
    print('fir2')
    GPIO.output(m11, True)#turn left
    GPIO.output(m10, False)
    GPIO.output(m21, False)
    GPIO.output(m20, False)
    time.sleep(0.2)
    drive_motor('forward')

def b_ir1_detect(channel3):##left sensor
    print('bir1')
    GPIO.output(m11, False)#turn left
    GPIO.output(m10, False)
    GPIO.output(m21, True)
    GPIO.output(m20, False)
    time.sleep(0.2)
    drive_motor('backward')
        
def b_ir2_detect(channel4):##right sensor
    print('bir2')
    GPIO.output(m11, False)#turn right
    GPIO.output(m10, True)
    GPIO.output(m21, False)
    GPIO.output(m20, False)
    time.sleep(0.2)
    drive_motor('backward')

reset = True

while True:
    try:
        drive_motor('stop')
        sound = False
        main = True
        facedetect = False
        print(' Women Safety Night Patrolling Robot')
        lcd.clear()
        lcd.message('  Women Safety \nPatrolling Robot')
        time.sleep(3)
        print('Searching for Camera')
        lcd.clear()
        lcd.message('Searching for\nCameras...')
        time.sleep(1)

        camera1 = cv2.VideoCapture(0)
        camera2 = cv2.VideoCapture(1)

        if not camera1.isOpened():
                main = False
                print ("can't open the camera1")
                lcd.clear()
                lcd.message('Error:Camera1 not\nFound')
                time.sleep(2)
                lcd.clear()
                lcd.message('Connect Camera &\npress reset')
                while GPIO.input(button) == True:
                    main = False
                    None
        else:
                main = True
                print ("camera1 found")
                lcd.clear()
                lcd.message('Camera1 Found')
                time.sleep(2)
                if not camera2.isOpened():
                        main = False
                        print ("can't open the camera2")
                        lcd.clear()
                        lcd.message('Error:Camera2 not\nFound')
                        time.sleep(2)
                        lcd.clear()
                        lcd.message('Connect Camera &\npress reset')
                        cam2 = False
                        while GPIO.input(button) == True:
                            main = False
                            None
                else:
                        main = True
                        print ("camera2 found")
                        lcd.clear()
                        lcd.message('Camera2 Found')
                        time.sleep(2)

        if main:
            print('connecting to internet')
            lcd.clear()
            lcd.message('Connecting to \nInternet....')
            time.sleep(1)
            t1 = datetime.now()
            while not check_connectivity() :
                    t2 = datetime.now()
                    delta = t2 - t1
                    time_elapse = delta.total_seconds()
                    if time_elapse > 10:
                        lcd.clear()
                        lcd.message('Error: Check\nInternet Connection ')
                        print ("error check you internet connection")
                        time.sleep(2)
                        main = False
                        while GPIO.input(button) == True:
                            lcd.clear()
                            lcd.message('Press Reset to\nRestart')
                            time.sleep(0.5)
                        break
                    else:
                        main = True
            if main:
                lcd.clear()
                lcd.message('Connected...')
                time.sleep(1)
            
        while (main == True) and (GPIO.input(button) == True):
            lcd.clear()
            lcd.message('Monitoring...')
            distance1 = distance_isr(front_trigg, front_echo)
            distance2 = distance_isr(back_trigg, back_echo)
            while sound == False and GPIO.input(button) == True:
                direct = check_sound()
                time.sleep(0.5)
                if direct == 'forward' or direct == 'backward':
                    lcd.clear();
                    lcd.message('Direction :\n' + str(direct))
                    time.sleep(1)
                    print('direction : ' + str(direct))
                    sound = True

            drive_motor(direct)
            if direct == 'forward' and distance1 > 20:
                lcd.clear()
                lcd.message('Forward')
                time.sleep(1)
                GPIO.add_event_detect(f_ir1, GPIO.BOTH, callback = f_ir1_detect, bouncetime=1)
                GPIO.add_event_detect(f_ir2, GPIO.BOTH, callback = f_ir2_detect, bouncetime=1)
                GPIO.remove_event_detect(b_ir1)
                GPIO.remove_event_detect(b_ir2)
                while facedetect == False:
                    lcd.clear()
                    lcd.message('Searching Face')
                    time.sleep(1)
                    face = detect_face(camera1)
                    if face:
                        print('Face detected..\nSending data..')
                        lcd.clear()
                        lcd.message('Face detected..\nSending data..')
                        drive_motor('stop')
                        time.sleep(2)
                        send_iot_data()
                        facedetect = True
                    distance = distance_isr(front_trigg, front_echo)
                    print('distance: ', str(distance)) 
                    if distance < 20:
                        drive_motor('stop')
                        send_iot_data()
                        break
                    if GPIO.input(button) == False:
                        break
        
            if direct == 'backward' and distance2 > 20:
                GPIO.add_event_detect(b_ir1, GPIO.BOTH, callback = b_ir1_detect, bouncetime=1)
                GPIO.add_event_detect(b_ir2, GPIO.BOTH, callback = b_ir2_detect, bouncetime=1)
                GPIO.remove_event_detect(f_ir1)
                GPIO.remove_event_detect(f_ir2)
                while facedetect == False:
                    lcd.clear()
                    lcd.message('Searching Face')
                    face = detect_face(camera2)
                    if face:
                        lcd.clear()
                        lcd.message('Face detected..\nSending data..')
                        drive_motor('stop')
                        time.sleep(2)
                        send_iot_data()
                        facedetect = True
                    distance = distance_isr(back_trigg, back_echo)
                    print('distance: ', str(distance))
                    if distance < 20:
                        drive_motor('stop')
                        send_iot_data()
                        break
                    if GPIO.input(button) == False:
                        break
            direct = ''
            sound = False
            facedetect = False
            drive_motor('stop')
            GPIO.remove_event_detect(b_ir1)
            GPIO.remove_event_detect(b_ir2)
            GPIO.remove_event_detect(f_ir1)
            GPIO.remove_event_detect(f_ir2)
        time.sleep(1)
        camera1.release()
        camera2.release()
    except:
        GPIO.remove_event_detect(b_ir1)
        GPIO.remove_event_detect(b_ir2)
        GPIO.remove_event_detect(f_ir1)
        GPIO.remove_event_detect(f_ir2)
        direct = ''
        sound = False
        facedetect = False
        drive_motor('stop')
        camera1.release()
        camera2.release()
        lcd.clear()
        lcd.message('Got Error!!!')
        time.sleep(2)
        while GPIO.input(button) == True:
            lcd.clear()
            lcd.message('Press Reset to\nRestart')
            time.sleep(0.5)

