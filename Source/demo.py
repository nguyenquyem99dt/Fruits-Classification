import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import time
import argparse
import matplotlib.pyplot as plt

'''CLASS = ['Granadilla', 'Salak', 'Apple Red 2', 'Strawberry Wedge', 'Tangelo', 'Banana', 'Tamarillo', 'Apple Golden 1', 'Passion Fruit', 'Cantaloupe 2', 'Pitahaya Red', 'Physalis', 'Grape Pink', 'Mango', 'Papaya', 'Apple Red Yellow', 'Apple Granny Smith', 'Maracuja', 'Huckleberry', 'Apricot', 'Pomegranate', 'Kiwi', 'Grape White 2', 'Avocado ripe', 'Melon Piel de Sapo', 'Carambula', 'Apple Red Delicious', 'Apple Golden 3', 'Mulberry', 'Apple Braeburn', 'Banana Red', 'Grapefruit Pink', 'Lychee', 'Cherry Rainier', 'Limes', 'Raspberry', 'Strawberry', 'Cantaloupe 1', 'Pear', 'Lemon Meyer', 'Pear Abate', 'Nectarine', 'Apple Golden 2', 'Dates', 'Pear Williams', 'Rambutan', 'Pineapple Mini', 'Physalis with Husk', 'Mandarine', 'Kaki', 'Apple Red 3', 'Avocado', 'Cocos', 'Pineapple', 'Cactus fruit', 'Pepino', 'Cherry 1', 'Quince', 'Lemon', 'Peach Flat', 'Orange', 'Cherry 2', 'Grape White', 'Apple Red 1', 'Kumquats', 'Grapefruit White', 'Clementine', 'Guava', 'Pear Monster', 'Plum', 'Peach']

IMG_SIZE=64
img_test = cv2.imread('kiwi.jpg')
img_test = cv2.resize(img_test, (IMG_SIZE, IMG_SIZE))

img_test_array = np.array(img_test).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
trained_model = load_model('model.h5')

predicted_class = np.argmax(trained_model.predict(img_test_array))
print('Predicted label: ', CLASS[predicted_class])

img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)
plt.imshow(img_test)
plt.show()'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    # Command line options
    parser.add_argument('--model', type=str, 
    help = 'Path to model')

    parser.add_argument('--input', type=str,
    help = 'Path to input')

    parser.add_argument('--image', default=False,
    help='Image detection mode')

    FLAGS = parser.parse_args()

    # loading the stored model from file
    model=load_model(FLAGS.model)

    IMG_SIZE = 64
    CLASS = ['Granadilla', 'Salak', 'Apple Red 2', 'Strawberry Wedge', 'Tangelo', 'Banana', 'Tamarillo', 'Apple Golden 1', 'Passion Fruit', 'Cantaloupe 2', 'Pitahaya Red', 'Physalis', 'Grape Pink', 'Mango', 'Papaya', 'Apple Red Yellow', 'Apple Granny Smith', 'Maracuja', 'Huckleberry', 'Apricot', 'Pomegranate', 'Kiwi', 'Grape White 2', 'Avocado ripe', 'Melon Piel de Sapo', 'Carambula', 'Apple Red Delicious', 'Apple Golden 3', 'Mulberry', 'Apple Braeburn', 'Banana Red', 'Grapefruit Pink', 'Lychee', 'Cherry Rainier', 'Limes', 'Raspberry', 'Strawberry', 'Cantaloupe 1', 'Pear', 'Lemon Meyer', 'Pear Abate', 'Nectarine', 'Apple Golden 2', 'Dates', 'Pear Williams', 'Rambutan', 'Pineapple Mini', 'Physalis with Husk', 'Mandarine', 'Kaki', 'Apple Red 3', 'Avocado', 'Cocos', 'Pineapple', 'Cactus fruit', 'Pepino', 'Cherry 1', 'Quince', 'Lemon', 'Peach Flat', 'Orange', 'Cherry 2', 'Grape White', 'Apple Red 1', 'Kumquats', 'Grapefruit White', 'Clementine', 'Guava', 'Pear Monster', 'Plum', 'Peach']
    
    # image mode
    if FLAGS.image:
        try:
            image = cv2.imread(FLAGS.input) # read image
            orig = image.copy() # copy image
            
        except:
            print('Can not read input image. Please check path again!')
        
        else:            
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) # resize image to fit with model 
            image = image.astype("float") / 255.0 # convert to float
            image = img_to_array(image) # convert image to array
            image = np.expand_dims(image, axis=0) # expend dims
                
            predicted_class = np.argmax(model.predict(image)) 

            label = 'Predict: ' + str(CLASS[predicted_class])
            print(label)

            orig = cv2.resize(orig, (600, 600))
            cv2.putText(orig, label, (10, 20),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 255, 255), 6) 
            cv2.putText(orig, label, (10, 20),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
            
            cv2.imwrite('prediction.jpg', orig) # save output image

            cv2.imshow("Output", orig) # show output image
            cv2.waitKey(0) # wait key
            cv2.destroyAllWindows()

    # video mode
    else: 
        cap = cv2.VideoCapture(FLAGS.input)
        time.sleep(2)

        if cap.isOpened(): # try to get the first frame
            rval, frame = cap.read()
        else:
            rval = False

        while(1):
            rval, image = cap.read()

            if rval==True:
                orig = image.copy() # copy image
            
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) # resize image to fit with model
                image = image.astype("float") / 255.0 # convert to float type
                image = img_to_array(image) # convert image to array
                image = np.expand_dims(image, axis=0) # expand dims
                
                tic = time.time() # Start time
                predicted_class = np.argmax(model.predict(image)) 
                label = 'Predict: ' + str(CLASS[predicted_class])
                toc = time.time() # End time

                fps = 1 / np.float64(toc - tic) # Calculate fps

                # print some information on console
                print("Time taken = ", toc - tic)
                print("FPS: ", fps)
                print('--------------------------------')


                cv2.putText(orig, label, (10, 20),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 255, 255), 6)
                cv2.putText(orig, label, (10, 20),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)#
                
                cv2.putText(orig, 'FPS: '+ str(fps), (10, 40),  cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 6)
                cv2.putText(orig, 'FPS: '+ str(fps), (10, 40),  cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 240, 0), 2) # put fps
                
                cv2.imshow("Output", orig) # show output image
                key = cv2.waitKey(1) # wait least 1ms

                if key == 27: # exit on ESC
                    break
            elif rval==False: # If no frame to read -> break
                    print('No frame to read!')
                    break
    
        cap.release()
        cv2.destroyAllWindows()