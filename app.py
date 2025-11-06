from transformers import pipeline
import os
import pandas as pd
"""

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

genderMap = {0: "male", 1: "female"}
raceMap = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Other"}
pipe = pipeline("image-classification", model="dima806/fairface_age_image_detection")

def getAgeMidpoint(ageLabel):
    if ageLabel == "more than 70":
        return 85
    parts = ageLabel.split("-")
    if len(parts) == 2:
        return (int(parts[0]) + int(parts[1])) / 2
    return int(parts[0])

data = []
directory = './data/part1/'
i = 1
for name in os.listdir(directory):
    print(f"Processing image {i}/{len(os.listdir(directory))}")
    i += 1
    parts = name.split("_")
    if not len(parts) >= 4:
        continue
    age = int(parts[0])
    gender = int(parts[1])
    race = int(parts[2])

    predictedAge = pipe(os.path.join(directory, name))[0]['label']
    data.append({
        "filename": name,
        "age": age,
        "gender": genderMap[gender],
        "race": raceMap[race],
        "predicted_age": getAgeMidpoint(predictedAge)
    })

df = pd.DataFrame(data)
print(df)
df.to_csv("output.csv", index=False)