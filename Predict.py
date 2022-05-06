import os
import cv2
from tensorflow.keras.models import load_model
from numpy import expand_dims
from numpy import load
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from datetime import datetime, timedelta
import mysql.connector
import numpy as np # linear algebra


#loading haarcascade_frontalface_default.xml
face_model = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
modelm = load_model('models/masknet1.h5')
modelf = load_model('models/facenet_keras.h5')
img_height = 224
img_width = 224
required_size=(160, 160)

mask_label = {0:'MASK',1:'NO MASK'}
dist_label = {0:(0,255,0),1:(255,0,0)}

webcam = cv2.VideoCapture(0)

# load faces
data = load('models/Dataset.npz')
testX_faces = data['arr_2']

# load face embeddings
data = load('models/Dataset-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)

# fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)


# get the face embedding for one face
def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  #database="mydatabase"
)

#print(mydb)
mycursor = mydb.cursor()
mycursor.execute("CREATE DATABASE IF NOT EXISTS z_intelligence")
mycursor.execute("USE z_intelligence")
mycursor.execute("CREATE TABLE IF NOT EXISTS Entry_Exit_Logs(EntryId INT NOT NULL AUTO_INCREMENT, EmployeeID VARCHAR(20) NOT NULL, EntryDate DATE NOT NULL, EntryTime TIME NOT NULL, ExitDate DATE NOT NULL, ExitTime TIME NOT NULL, PRIMARY KEY (EntryId))")


dicta = {}
dictl = {}
checktime=20
sendtime=5
updatetime = datetime.now();
webcam = cv2.VideoCapture(0)
while True:
    (rval, im) = webcam.read()
    img=cv2.flip(im,1,1) #Flip to act as a mirror
    faces = face_model.detectMultiScale(img,scaleFactor=1.1, minNeighbors=4) #returns a list of (x,y,w,h) tuples
    new_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #colored output image

    for i in range(len(faces)):
        (x,y,w,h) = faces[i]
        
        crop = new_img[y:y+h,x:x+w]
        crop1 = cv2.resize(crop,(img_height,img_width))
        crop1 = np.reshape(crop1,[1,img_height,img_width,3])/255.0
        mask_result = modelm.predict(crop1)
        (mask, withoutMask) = mask_result[0]
        #flabel = "{}: {:.2f}%".format(mask_label[mask_result.argmax()], max(mask, withoutMask) * 100)
        #cv2.putText(new_img,flabel,(x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0) if mask > withoutMask else (0,0,255),2)
        #cv2.rectangle(new_img,(x,y),(x+w,y+h),(0,255,0) if mask > withoutMask else (0,0,255) ,2)
        
        crop2 = cv2.resize(crop,(160,160))
        random_face_pixels =crop2
        random_face_emb = get_embedding(modelf, random_face_pixels)
        samples = expand_dims(random_face_emb, axis=0)
        yhat_class = model.predict(samples)
        yhat_prob = model.predict_proba(samples)
        class_index = yhat_class[0]
        class_probability = yhat_prob[0,class_index] * 100
        #class_probability = int(class_probability)
        #print(type(class_probability))
        predict_names = out_encoder.inverse_transform(yhat_class)
        #print('Predicted: %s (%.2f)' % (predict_names[0], class_probability))
        if(mask_label[mask_result.argmax()]=="NO MASK" and (max(mask, withoutMask) * 100)>85):
            if((int(class_probability))>=99):
                flabel = 'Predicted: %s (%.2f)' % (predict_names[0], class_probability)
                if(predict_names[0] not in dicta):
                    print("Created")
                    dicta[predict_names[0]]=datetime.now();
                    dictl[predict_names[0]]=datetime.now();
                elif(predict_names[0] in dicta):
#                     diff =datetime.now() - dictl.get(predict_names[0])
#                     if(diff.seconds>checktime):
#                         print("Updated Both")
#                         dicta[predict_names[0]]=datetime.now();
#                         dictl[predict_names[0]]=datetime.now();
#                     else :
                        print("Updated Left")
                        dictl[predict_names[0]]=datetime.now();
            else:
                flabel = 'Predicted: UNKNOWN (%.3f)' % (class_probability)
            
        else:
            flabel = "{}: {:.2f}%".format(mask_label[mask_result.argmax()], max(mask, withoutMask) * 100)
        
        cv2.putText(new_img,flabel,(x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
        cv2.rectangle(new_img,(x,y),(x+w,y+h),(255,0,0),1)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR) #colored output image
    delete = []
    u_diff=datetime.now()-updatetime
    if(u_diff.seconds>sendtime):
        for id in dictl:
            check = datetime.now()-dictl.get(id)
            dayl = dictl.get(id).day
            daya = dicta.get(id).day
            monthl = dictl.get(id).month
            montha = dicta.get(id).month
            yearl = dictl.get(id).year
            yeara = dicta.get(id).year
            hourl = dictl.get(id).hour
            minutel = dictl.get(id).minute
            secondl = dictl.get(id).second
            houra = dicta.get(id).hour
            minutea = dicta.get(id).minute
            seconda = dicta.get(id).second
            
            EntryDate = str(yeara) +"-"+ str(montha) +"-"+ str(daya)
            ExitDate = str(yearl) +"-"+ str(monthl) +"-"+ str(dayl)
            
            EntryTime = str(houra) +":"+ str(minutea) +":"+ str(seconda)
            ExitTime = str(hourl) +":"+ str(minutel) +":"+ str(secondl)
            
            if(check.seconds>checktime):
                query = "INSERT INTO entry_exit_logs(EmployeeID,EntryDate,EntryTime,ExitDate,ExitTime) VALUES('"+ str(id) +"','"+ EntryDate +"','"+ EntryTime +"','"+ ExitDate +"','"+ ExitTime +"');"
                mycursor.execute(query)
                print("Added Entry of Person with last exit time(stored in dictl) and entry time(stored in dicta)")
                delete.append(id)
        for id in delete:
            dictl.pop(id)
            dicta.pop(id)
        updatetime = datetime.now();
        print(updatetime)

        
    cv2.imshow('LIVE',   new_img)
    #time.sleep(2)
    key = cv2.waitKey(10)
    # if Esc key is press then break out of the loop 
    if key == 27: #The Esc key
        break
# Stop video
webcam.release()

# Close all started windows
cv2.destroyAllWindows()
mycursor.execute("COMMIT;")
mycursor.close()
mydb.close()