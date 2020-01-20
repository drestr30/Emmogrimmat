import os
from shutil import copyfile


base_path ="/home/fourier/Documentos/Datasets_emociones/CK_DB/CK+"
out_path="/home/fourier/Documentos/Datasets_emociones/CK_ordered"


def get_emotion_label(path):
    ck_labels = {0: "neutral", 1: "anger", 2: "contempt", 3: "disgust", 4: "fear", 5: "happy", 6: "sadness",
                 7: "surprise"}

    emotion_file = os.listdir(path)[0]

    f = open(os.path.join(path, emotion_file), "r")
    number = f.readline()
    return ck_labels[int(float(number))]

subjects_path = os.path.join(base_path, "cohn-kanade-images")
for subject in os.listdir(subjects_path):
    if not subject.startswith("."):
        for sequence in os.listdir(os.path.join(subjects_path, subject)):
            if not sequence.startswith("."):

                try:
                    emotion = get_emotion_label(os.path.join(base_path,"Emotion", subject,sequence))
                    print(emotion)
                except IndexError:
                    print("No emotion label")
                    break
                except FileNotFoundError:
                    print("No emotion label folder")

                img_list = sorted(os.listdir(os.path.join(subjects_path, subject, sequence)))
                emotion_img_num = int(len(img_list) * 0.3)

                if not os.path.exists(os.path.join(out_path,emotion)):
                    os.mkdir(os.path.join(out_path, emotion))

                for ind in range(-1,-1*emotion_img_num-2,-1): #Toma el ultimo 30% de las imagenes
                    img_path = os.path.join(subjects_path, subject, sequence, img_list[ind])
                    order_path = os.path.join(out_path,emotion, img_list[ind])
                    copyfile(img_path,order_path)



