from pymongo import MongoClient
import pickle
import pymongo
from dotenv import load_dotenv
import os

class ImageDatabase:
    # here database is static variable
    database = None
    
    @staticmethod
    def get_database_instance():
        if ImageDatabase.database == None:
            ImageDatabase.database = ImageDatabase()
        return ImageDatabase.database

    def __init__(self, database_name='image_database', collection_name='image_collection'):
        load_dotenv()
        self.client = MongoClient(os.getenv("API_ENDPOINT"), 27017)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]

    def add_image(self, image, label):
        data = {"image": image,'label': label}
        data['image'] = pickle.dumps( image, protocol=2)

        self.collection.insert_one(data)

    def remove_image(self, image):
        self.collection.delete_one({'image': image})

    def generate_label_hashmap(self):
        label_hashmap = {}
        cursor = self.collection.find()

        for document in cursor:
            label = document['label']
            image = pickle.loads(document['image'])

            if label not in label_hashmap:
                label_hashmap[label] = [image]
            else:
                label_hashmap[label].append(image)

        return label_hashmap

    def close_connection(self):
        self.client.close()

def get_images_from_dir():
    import os
    import cv2 as cv

    images = []
    labels = {}
    currID = 0
    train_image_dimensions = (152, 152)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    IMG_DIR = os.path.join(BASE_DIR, 'images')

    for root, dirs, files in os.walk(IMG_DIR):
        for filer in files:
            if filer.endswith('png') or filer.endswith('jpg'):
                path = os.path.join(root, filer)
                label = os.path.basename(root).replace(' ', '_').lower()
                if label not in labels.values():
                    labels[label] = currID
                    currID += 1
                image = cv.imread(path)
                image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  
                resized_img = cv.resize(image, train_image_dimensions, interpolation=cv.INTER_LINEAR)
                images.append([resized_img, label])


    return images


def populate_database(db, collection):
    for image in collection:
        db.add_image(image[0], image[1])

# Example usage:
if __name__ == "__main__":
    image_db = ImageDatabase()

    # Add images
    img_list = get_images_from_dir()
    print(img_list[0][0])
    
    image_db.add_image(img_list[0][0], img_list[0][1])
    
    # populate_database(image_db, img_list)

    # Generate and print the label hashmap
    # label_hashmap = image_db.generate_label_hashmap()
    # print("Label Hashmap:")
    # print(label_hashmap)

    # # Generate and print the updated label hashmap
    # updated_label_hashmap = image_db.generate_label_hashmap()
    # print("\nUpdated Label Hashmap:")
    # print(updated_label_hashmap)

    # Close the MongoDB connection
    image_db.close_connection()
