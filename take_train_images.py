from pynput.keyboard import Key, Listener
from picamera import PiCamera
from time import sleep
from datetime import datetime
import ftplib
import glob
import os

camera = PiCamera(resolution=(300,300))

# Wait for the automatic gain control to settle
sleep(2)

def uploadFiles():
    
    try:
        ftp = ftplib.FTP('ip')
    except:
        print("Connection error.")
    else:
        ftp.login('uploader','myData20')
        ftp.cwd('/pi3_images')

        for filepath in glob.iglob('images/*.jpeg'):
            head, tail = os.path.split(filepath)
            print(head, tail)
            file = open(filepath,'rb')
            ftp.storbinary('STOR '+tail, file)
            os.remove(filepath)
        ftp.quit()

def on_release(key):
    if key == Key.esc: # Stop listener
        return False
    if key == Key.space:
        date_time = datetime.now().strftime("%Y%m%d%H%M%S")
        camera.capture("images/" + date_time + '.jpeg', format="jpeg")
        print('Picture taken at ' + date_time)
        uploadFiles()
        
# Collect events until released
with Listener(
        on_release=on_release) as listener:
    listener.join()
    