from model import model_train, model_predict
import numpy as np

MODEL = None
CRIMINAL_LIST = {}

def convert_data(data):
    import cv2
    import numpy as np

    # Inside the process_image function

    print(data, type(data))

    nparr = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cascade = cv.CascadeClassifier(CAS_DIR+'haarcascade_frontalface_alt2.xml')
    faces = cascade.detectMultiScale(image, minNeighbors=1, scaleFactor=1.5)

    for x, y, w, h in faces:
        roi_gray = gray[y: y + h, x: x + w]
        roi_gray = cv.resize(roi_gray, train_image_dimensions, interpolation=cv.INTER_LINEAR)
        roi_gray = roi_gray.reshape(-1, 152, 152, 1) / 255
        
        return roi_gray
    


def generate_result(data, model):
    global MODEL
    global CRIMINAL_LIST

    MODEL = model


    # the data is the image
    prediction = model_predict(data, MODEL)
    prediction = np.argmax(prediction)
    return prediction

def get_database_instance():
    import database
    return database.ImageDatabase.get_database_instance()

def create_training_data(data):
    global CRIMINAL_LIST

    x = np.array([])
    y = np.array([])

    currID = 0
    for label in data:
        CRIMINAL_LIST[currID] = label
        for image in data[label]:
            x = np.append(x, image)
            y = np.append(y, currID)
        
        currID += 1
    return x, y

def create_train_data(data):
    global CRIMINAL_LIST
    

    x = np.array([])
    y = np.array([])

    rev_list = {}

    currID = 0
    
    for img in data:
        if img[1] not in rev_list:
            CRIMINAL_LIST[currID] = img[1]
            rev_list[img[1]] = currID
            currID += 1

        x = np.append(x, img[0])
        y = np.append(y, rev_list[img[1]])

    return x, y, currID


def train_model():
    from database import get_images_from_dir

    print("IN TRAIN MODEL")
    global MODEL
    global CRIMINAL_LIST
    
    # db = get_database_instance()

    print("INSTANCE CREATED")
    images = get_images_from_dir()

    # data = db.generate_label_hashmap()

    print("DATA GENERATED")

    X, y, output = create_train_data(images)

    print("TRAINING DATA CREATED")

    MODEL = model_train(X, y, MODEL, output)

    return "SUCCESS", CRIMINAL_LIST, MODEL