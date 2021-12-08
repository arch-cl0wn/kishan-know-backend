# SPDX-FileCopyrightText: 2021 ladyada for Adafruit Industries
# SPDX-License-Identifier: MIT

import time
import busio
import board
import adafruit_amg88xx
from firebase import firebase
import data_checker as dc

i2c = busio.I2C(board.SCL, board.SDA)
amg = adafruit_amg88xx.AMG88XX(i2c)
sum1,c1,c,finval=0,0,0,0

firebase = firebase.FirebaseApplication('https://kishanknow-8a73c-default-rtdb.asia-southeast1.firebasedatabase.app/', None)

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
    dc.call_func()
    time.sleep(120)