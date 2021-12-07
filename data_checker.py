import Adafruit_AMG88xx_thermalcamtest as Adafruit
import csv

with open('microbedatasetcsv.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if Adafruit.diff>=row["MTD_initial"] and Adafruit.diff<=row["MTD_final"]:
            print(row["Microbe_Name"])

