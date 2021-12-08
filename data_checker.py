import Adafruit_AMG88xx_thermalcamtest as Adafruit
import requests

def call_func():
    with open('/home/pi/Adafruit_CircuitPython_AMG88xx/examples/data.txt','r') as f:
        l=f.readlines()
        for row in l:
            words=row.strip().split(",")
            if (Adafruit.diff>=float(words[1])) and (Adafruit.diff<=float(words[2])):
                print(words[0])
                r = requests.post('https://maker.ifttt.com/trigger/Microbe_Detection/with/key/FhIUKYtS7ymqOC8ePCE0BN8Le2Zp7Rd4C2UFAzZsQl', params={"value1":words[0],"value2":words[3],"value3":"none"})

