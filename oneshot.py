from keras.layers import Dense, Input, Lambda
from keras.models import Sequential, Model, load_model
from keras import backend as K
import numpy as np
import random
import cv2, os

"""

Sample Image Stucture:
    ./samples(folder)
        - label1(folder)
            - /image1.jpg
            - /image2.jpg
            - /image3.jpg
            - /image4.jpg
            - ......
        - label2(folder)
            - /image1.jpg
            - /image2.jpg
            - /image3.jpg
            - /image4.jpg
            - ......
        - label3(folder)
            - /image1.jpg
            - /image2.jpg
            - /image3.jpg
            - /image4.jpg
            - ......
        - ......

"""

class oneShot:
    
    def __init__(self, learning_rate=0.0003, conf_threshold=0.7, 
                 facenet_path="model-weights/facenet-keras.h5"):
        print("\rLoading FaceNet Model", end="")
        self.facenet      = load_model(facenet_path)
        print("\r Loaded FaceNet Model")
        print("\rLoading OneShot Model", end="    ")
        self.model        = self.siameseModel()
        print("\r Loaded OneShot Model")
        self.num_dataset  = 0
        self.CONF_THRESH  = conf_threshold


    def siameseModel(self):
        # Input Layers
        left_input = Input((128,))
        right_input = Input((128,))
        # Fully Connected Network
        model = Sequential()
#        model.add(Dense(4096, activation='relu'))
#        model.add(Dense(2048, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        # Generate the encodings (feature vectors) for the two images
        encoded_l = model(left_input)
        encoded_r = model(right_input)    
        # Add a customized layer to compute the absolute difference between the encodings
        L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([encoded_l, encoded_r])    
        # Add a dense layer with a sigmoid unit to generate the similarity score
        prediction = Dense(1024, activation='relu'   )(L1_distance)
        prediction = Dense( 512, activation='relu'   )(prediction)
        prediction = Dense(   1, activation='sigmoid')(prediction)
        # Connect the inputs with the outputs
        siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
        siamese_net.compile(loss="binary_crossentropy",optimizer="adam",
    													metrics=["accuracy"])
        return siamese_net


    # Function to Classify the Output or Change the String to Binary
    def oneHotEnc(self, targetString):
        classes = np.unique(targetString.copy())
        Y = np.zeros((len(targetString), len(classes)))
        for i in range(len(targetString)): 
            Y[i][np.where(classes==targetString[i])[0]] = 1
        return Y

    
    # Create Dataset to Train the Model
    def createDataset(self, images, labels, forEach):
        inputs1 = []
        inputs2 = []
        outputs = []
        images = self.facenet.predict(images)
        for i in range(len(images)):
            _1 = images[i]
            pos = random.choice(range(len(images)))
            _2 = images[pos]
            for _ in range(int(forEach)):
                while labels[i] == labels[pos]:
                    pos = random.choice(range(len(images)))
                _2 = images[pos]
                inputs1.append(_1)
                inputs2.append(_2)
                outputs.append(0)
                while labels[i] != labels[pos]:
                    pos = random.choice(range(len(images)))
                _2 = images[pos]
                inputs1.append(_1)
                inputs2.append(_2)
                outputs.append(1)
        self.num_dataset = len(inputs1)
        inputs1  = np.array(inputs1)
        inputs2  = np.array(inputs2)
        outputs  = np.array(outputs)
        return inputs1, inputs2, outputs


    # Train the Model and can Save the Weights
    def trainModel(self, X, Y, batch_size=32, epochs=5, save_location=None, validation_split=0.2, forEach=2):
        X1, X2, Y = self.createDataset(X, Y, forEach=forEach)
        self.model.fit([X1, X2], Y, batch_size=batch_size, validation_split=validation_split, epochs=epochs)
        if save_location:
            self.model.save_weights(os.path.join(save_location, "model_siamese.h5"))
        return None
    
    
    # Load the Model with Pre-trained Weights
    def loadModel(self, load_location):
        self.model.load_weights(os.path.join(load_location, "model_siamese.h5"))
        return None
        
    
    # Find the Prediction with Each Samples
    def findConfidence(self, face, samples_location):
        confidences = []
        labels      = []
        for label in os.listdir(samples_location):
            for image in os.listdir(os.path.join(samples_location, label)):
                sample_image = cv2.resize(cv2.imread(os.path.join(samples_location, label, image)), (160, 160))
                sample_image = self.facenet.predict(np.array([sample_image]))
                confidences.append(self.model.predict([face, sample_image])[0][0])
                labels.append(label)
        return confidences, labels


    # Return the Recognized Face name
    def detectFace(self, image, samples_location):
        face = self.facenet.predict(np.array([image]))
        confidences, labels = self.findConfidence(face, samples_location)
        if max(confidences) > self.CONF_THRESH:
            return labels[confidences.index(max(confidences))]
        else:
            return "UNKNOWN"
        
    


"""

#Try this first commented section


samples_location="Z:/Python/Attendance System/samples"

model = oneShot()
X, Y = [], []
for label in os.listdir("dataset/"):
    for image in os.listdir(os.path.join("dataset", label)):
        X.append(np.array(cv2.resize(cv2.imread(os.path.join("dataset", label, image)), (160,160))))
        Y.append(label)
X = np.array(X)
Y = np.array(Y)

model.trainModel(X, Y, epochs=2, validation_split=0.1, forEach=2)

model.detectFace(X[46], samples_location=samples_location)

model.findConfidence(model.facenet.predict(np.array([X[46]])), samples_location)


face = model.facenet.predict(np.array([[X[10]]]))
confidences = []
labels      = []
for label in os.listdir(samples_location):
    for image in os.listdir(os.path.join(samples_location, label)):
        sample_image = cv2.resize(cv2.imread(os.path.join(samples_location, label, image)), (160, 160))
        sample_image = model.facenet.predict(np.array([sample_image]))
        confidences.append(model.model.predict([face, sample_image])[0][0])
        labels.append(label)




model.findConfidence(np.array([model.facenet.predict(np.array([X[10]]))]), samples_location="Z:/Python/Attendance System/samples")


model.facenet.predict(np.array([X[1]]))

dataset = model.createDataset(X, Y)

model.model.fit([dataset[0][:,0], dataset[0][:,1]], dataset[1], epochs=10)











"""
