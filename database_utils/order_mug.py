import glob, os, os.path
import operator
from shutil import copyfile
import fnmatch

path = "/home/fourier/Documentos/Datasets_emociones/MUG_DB/subjects3"
out_path = "/home/fourier/Documentos/Datasets_emociones/MUG_ordered"
count = 0
for subject in os.listdir(path):
    if not subject.startswith('.') and not subject == "RAR":
        for session in os.listdir(os.path.join(path, subject)):
            if "session" in session:
                if not session.startswith('.'):
                    for emotion in os.listdir(os.path.join(path, subject, session)):
                        if not emotion.startswith('.') and not emotion == "mixed" and not emotion == "extra" and emotion == "neutral" or emotion == "fear"or emotion == "sadness":
                            for take in os.listdir(os.path.join(path, subject, session, emotion)):
                                    if not take.startswith('.') and not take == "RAR":
                                        list = []
                                        for image in os.listdir(os.path.join(path, subject, session, emotion, take)):
                                            if not image.startswith('.') and "video" not in image and not image == "Thumbs.db":
                                                # if not os.path.splitext(image)[-1] == ".jpg":
                                                #     os.remove(os.path.join(path, subject, emotion, take, image))
                                                #print(os.path.splitext(image)[0].lower().split("_")[-1]), os.path.join(path,subject,session,emotion,take,image)
                                                list.append((int(os.path.splitext(image)[0].lower().split("_")[-1]), os.path.join(path,subject,session,emotion,take,image)))

                                        print("*************************")
                                        print(os.path.join(path,subject,session,emotion,take))
                                        list.sort(key=operator.itemgetter(0))
                                        file_index=int((list[-1][0]-list[0][0])/2)
                                        out_path_emotion = os.path.join(out_path, emotion)
                                        if not os.path.exists(out_path_emotion):
                                            os.mkdir(out_path_emotion)
                                        #os.rename(list[file_index][1], os.path.join(out_path, emotion, str(count) + ".jpg"))
                                        imgNumber = 7
                                        unt = range(file_index -imgNumber ,file_index + imgNumber,1)
                                        if emotion == "neutral":
                                            unt = range(1,len(list),1)
                                        for ind in unt:
                                            print(list[ind][1])
                                            copyfile(list[ind][1],os.path.join(out_path_emotion, subject+ emotion+ take+str(list[ind][0])+ ".jpg"))
                                            count = count + 1

            else:
                    for emotion in os.listdir(os.path.join(path, subject)):
                        if not emotion.startswith('.') and not emotion == "mixed" and not emotion == "extra" and emotion == "neutral" or emotion == "fear"or emotion == "sadness":
                            for take in os.listdir(os.path.join(path, subject, emotion)):
                                if not take.startswith('.') and not take == "RAR":
                                    list = []
                                    for image in os.listdir(os.path.join(path, subject, emotion, take)):
                                        if not image.startswith('.') and "video" not in image and not image == "Thumbs.db":
                                                # if not os.path.splitext(image)[-1] == ".jpg":
                                            # if not os.path.splitext(image)[-1] == ".jpg":
                                            #     os.remove(os.path.join(path, subject, emotion, take, image))
                                            #print(os.path.splitext(image)[0].lower().split("_")[-1]), os.path.join(path,
                                                                                                                    #subject,
                                                                                                                  # emotion,
                                                                                                                   #take,
                                                                                                                   #image)
                                            list.append((int(os.path.splitext(image)[0].lower().split("_")[-1]),
                                                         os.path.join(path, subject, emotion, take, image)))

                                    print("*************************")
                                    print(os.path.join(path, subject, emotion, take))
                                    list.sort(key=operator.itemgetter(0))
                                    file_index = int((list[-1][0] - list[0][0]) / 2)
                                    out_path_emotion = os.path.join(out_path, emotion)
                                    if not os.path.exists(out_path_emotion):
                                        os.mkdir(out_path_emotion)
                                    # os.rename(list[file_index][1], os.path.join(out_path, emotion, str(count) + ".jpg"))
                                    imgNumber = 7
                                    unt = range(file_index - imgNumber, file_index + imgNumber, 1)
                                    if emotion == "neutral":
                                        unt = range(1,len(list),1)
                                    for ind in unt:
                                        print(list[ind][1])
                                        copyfile(list[ind][1], os.path.join(out_path_emotion, subject+emotion+take+str(list[ind][0])+ ".jpg"))
                                        count = count + 1

