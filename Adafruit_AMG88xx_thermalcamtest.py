# SPDX-FileCopyrightText: 2021 ladyada for Adafruit Industries
# SPDX-License-Identifier: MIT

import time
import busio
import board
import adafruit_amg88xx
import requests
from firebase import firebase
print(9)

i2c = busio.I2C(board.SCL, board.SDA)
amg = adafruit_amg88xx.AMG88XX(i2c)
sum1,c1,c,finval=0,0,0,0

firebase = firebase.FirebaseApplication('https://kishanknow-8a73c-default-rtdb.asia-southeast1.firebasedatabase.app/', None)

def call_func():
    with open('/home/pi/Adafruit_CircuitPython_AMG88xx/examples/data.txt','r') as f:
        l=f.readlines()
        for row in l:
            words=row.strip().split(",")
            if (diff>=float(words[1])) and (diff<=float(words[2])):
                print(words[0])
                r = requests.post('https://maker.ifttt.com/trigger/Microbe_Detection/with/key/FhIUKYtS7ymqOC8ePCE0BN8Le2Zp7Rd4C2UFAzZsQl', params={"value1":words[0],"value2":words[3],"value3":"none"})



while True:
    for row in amg.pixels:
        # Pad to 1 decimal place
        for temp in row:
            sum1=sum1+float("{0:.1f}".format(temp))
            c=c+1
        finval=finval+sum1
        sum1=0
        c1=c1+c
        c=0
    res=finval/c1
    print(res)
    result = firebase.post('/temperature', res)
    result2=firebase.get('/farm_temperature',None)
    diff=result2-res
    result3=firebase.post('/Absolute_Temperature',diff)
    call_func()
    time.sleep(120)