import Adafruit_AMG88xx_thermalcamtest as Adafruit

with open('/home/pi/Adafruit_CircuitPython_AMG88xx/examples/data.txt','r') as f:
    l=f.readlines()
    for row in l:
        words=row.strip().split(",")
        if (Adafruit.diff>=float(words[1])) and (Adafruit.diff<=float(words[2])):
            print(words[0])

