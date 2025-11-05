from transformers import pipeline
import os
import pandas as pd
"""
pipe = pipeline("image-classification", model="dima806/fairface_age_image_detection")

print(pipe("./data/part1/96_1_0_20170110182515404.jpg"))
"""

"""
Label key
The labels of each face image is embedded in the file name, formated like [age]_[gender]_[race]_[date&time].jpg

    [age] is an integer from 0 to 116, indicating the age
    [gender] is either 0 (male) or 1 (female)
    [race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).
    [date&time] is in the format of yyyymmddHHMMSSFFF, showing the date and time an image was collected to UTKFace
"""



filename = "9_1_2_20161219190524395.jpg"

genderMap = {0: "male", 1: "female"}
raceMap = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Other"}

parts = filename.split("_")
age = int(parts[0])
gender = int(parts[1])
race = int(parts[2])
print(f"Age: {age}, Gender: {genderMap[gender]}, Race: {raceMap[race]}")

data = []
directory = './data/part1/'
for name in os.listdir(directory):
    print(directory+name)
    parts = name.split("_")
    if not len(parts) >= 3:
        continue
    age = int(parts[0])
    gender = int(parts[1])
    race = int(parts[2])

    data.append({
        "filename": name,
        "age": age,
        "gender": genderMap[gender],
        "race": raceMap[race]
    })

df = pd.DataFrame(data)
print(df)
