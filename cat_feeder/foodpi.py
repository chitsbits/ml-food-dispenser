# Once the button is pressed, the pi drives the Lego motor to dispense cat food
# press Ctrl-C to quit
import sys, getopt
import RPi.GPIO as GPIO
import time
from datetime import datetime


totalPushed = 0
print("Waiting for button: " ,totalPushed)

def button_pushed(channel):
    global totalPushed
    totalPushed += 1  
    print("pushed " + str(totalPushed))
    refill()
    
def setup_pin():
    #GPIO.cleanup()
    #time.sleep(1)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(13,GPIO.OUT)
    GPIO.setup(22,GPIO.OUT)
    GPIO.setup(23,GPIO.OUT)
    GPIO.setup(15,GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.add_event_detect(15,GPIO.RISING,callback=button_pushed)    
    time.sleep(1)
    
def motor_fwd():
    GPIO.output(22,GPIO.HIGH)
    time.sleep(0.01)
    GPIO.output(23,GPIO.LOW)
    time.sleep(0.01)
    
def motor_rev():
    GPIO.output(23,GPIO.HIGH)
    time.sleep(0.01)
    GPIO.output(22,GPIO.LOW)
    time.sleep(0.01)
    
def motor_stop():
    GPIO.output(22,GPIO.LOW)
    time.sleep(0.01)
    GPIO.output(23,GPIO.LOW)
    time.sleep(0.01)

def refill():
    GPIO.output(13,GPIO.HIGH)
    motor_fwd()
    time.sleep(5)
    motor_stop()    
    GPIO.output(13,GPIO.LOW)

def dispense():
    setup_pin()
    refill()
    GPIO.cleanup()

def main(argv):
    
    try:
      opts, args = getopt.getopt(argv,"ht")
    except getopt.GetoptError:
      print ('foodpi.py -t')
      sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print ('foodpi.py -t')
            print ('\t-t test')
            sys.exit()
        elif opt in ("-t"):
            print ('testing the motor')
            setup_pin()
            refill()
            GPIO.cleanup()
            print('cleanup done')
            sys.exit()


    #GPIO.cleanup()
    setup_pin()
    #GPIO.output(22,True)
    #GPIO.output(23,False)
    while True:
         try:
             now = datetime.now()
             print(str(now))
             time.sleep(0.5)
         except KeyboardInterrupt:
             
             print('\nCtrl-C detected, quitting...')
             break
             
    GPIO.cleanup()
    print("cleanup done")
    
if __name__ == "__main__":
   main(sys.argv[1:])
