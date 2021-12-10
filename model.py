import tensorflow as tf
import numpy as np
from PIL import Image

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        self.print_img = True
        self.testing = True

        self.batch_size = 1
        self.num_classes = 6 #(before 1800s, 1800-1900, 1900-1920, 1920-1940, 1940-1960, 1960-on )
        self.loss_list = []

        self.dense1size = 1000

        self.mnet2 = tf.keras.applications.MobileNetV2(include_top=False, alpha=.75, weights='imagenet', classes=self.num_classes, classifier_activation=None)
        self.dense1 = tf.keras.layers.Dense(self.dense1size, activation="sigmoid")
        self.dense2 = tf.keras.layers.Dense(self.num_classes, activation="softmax")
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(0.3)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    def call(self, inputs):
        inputs = inputs/255.0
        output = self.mnet2(inputs)
        output = self.flatten(output)    
        output = self.dropout(output)
        output = self.dense1(output)
        return self.dense2(output)

    def loss(self, probabilities, labels):
        labels = tf.one_hot(labels, 6, dtype=tf.int8)
        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, probabilities))
        return loss

    def accuracy(self, probabilities, labels):
        argmax = tf.argmax(probabilities, axis=-1)

        #Checks if the class with highest probability is the same as the true class
        correct_predictions = tf.equal(argmax, labels)

        #What portion of predictions are correct
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_images, train_dates):
    total_number_images_training = np.shape(train_images)[0]
    num_batches = total_number_images_training//model.batch_size

    #run one epoch
    for i in range(num_batches):
        print("train", i, num_batches)
        batch_images, batch_dates = get_batch(i, model.batch_size, train_images, train_dates)

        with tf.GradientTape() as tape:
            probabilities = model.call(batch_images)
            loss = model.loss(probabilities, batch_dates)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        model.loss_list.append(loss)

    return model.loss_list

def test(model, test_images, test_dates):
    probabilities = []
    accuracy = []
    total_number_images_testing = test_images.shape[0]

    num_batches = total_number_images_testing//model.batch_size

    for i in range(num_batches):
        print("test", i, num_batches)
        batch_images, batch_dates = get_batch(i, model.batch_size, test_images, test_dates)
        probability = model.call(batch_images)
        probabilities.append(probability)
        batch_accuracy = model.accuracy(probabilities, batch_dates)
        accuracy.append(batch_accuracy)

    #average accuracy across the batches
    return tf.reduce_mean(accuracy)

def main():
    print("running main")

    #read in train images, train dates, test images, test dates
    test_images = np.load('test_img.npy')
    test_dates = np.load('test_lab.npy')
    train_images = np.load('train_img.npy')
    train_dates = np.load('train_lab.npy')

    model = Model()

    num_epochs = 10 
    for epoch in range(num_epochs):
        print("epoch ", epoch)
        train(model, train_images, train_dates) 
        accuracy = test(model, test_images, test_dates)

        #write accuracy to file for each epoch for tuning 
        file = open("accuracy_by_epoch.txt", "a")
        string = "epoch = " + str(epoch) + ", accuracy = " + str(accuracy) + '\n'
        file.write(string)

    print("overall test accuracy is ", accuracy)


def get_batch(start_index, batch_size, images, dates):
    ending_index = start_index + batch_size
    return (images[start_index: ending_index], dates[start_index: ending_index])

main()
