from model import model_train, model_predict
import numpy as np
import cv2
import numpy as np

MODEL = None
CRIMINAL_LIST = {}

def convert_data(image):
    CAS_DIR = './utils/data/'
    cascade = cv2.CascadeClassifier(CAS_DIR+'haarcascade_frontalface_alt2.xml')
    colour = (0, 0, 255)
    train_image_dimensions = (152, 152)
    stroke = 2

    print("IN CONVERT DATA")

    image_np_array = np.array(image)
    image_col = cv2.cvtColor(image_np_array, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image_np_array, cv2.COLOR_BGR2GRAY)    
    
    # get faces from the image
    print("Image: ", image_gray.shape)
    print(image_np_array.shape)
    faces = cascade.detectMultiScale(image_gray, minNeighbors=5, scaleFactor=1.5)
    print("FACES: ", faces)

    for x, y, w, h in faces:
        print("IN LOOP")
        roi_gray = image_gray[y: y + h, x: x + w]
        roi_gray = cv2.resize(roi_gray, train_image_dimensions, interpolation=cv.INTER_LINEAR)
        roi_gray = roi_gray.reshape(-1, 152, 152, 1) / 255
        cv2.rectangle(image_col, (x, y), (x + w, y + h), colour, stroke)

        print("Shape: ", roi_gray.shape, data.shape)
        return roi_gray, image_col
    return None, image_col
    


def generate_result(data, model):
    global MODEL
    global CRIMINAL_LIST

    MODEL = model


    # the data is the image
    cutImage, predictionImage = convert_data(data)
    _, predictionImage = cv2.imencode('.jpg', predictionImage)
    if cutImage is None:
        return None, predictionImage

    prediction = model_predict(curImage, MODEL)
    print(len(prediction), prediction)
    prediction = np.argmax(prediction)
    print("Prediction: ", prediction)
    return prediction, predictionImage

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