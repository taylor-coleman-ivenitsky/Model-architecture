import tensorflow as tf
import numpy as np

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.batch_size = 100
        self.num_classes = 8 #(before 1700, 1700s, 1800s, 1900-1920, 1920-1940, 1940-60, etc. )
        self.loss_list = []

        #resize images to 224x224 because that is default input size for VGG
        self.VGG16 = tf.keras.applications.VGG16(include_top=False, classes=self.num_classes, classifier_activation=None)
        self.dense1 = tf.keras.layers.Dense(1000, activation="relu")
        self.dense2 = tf.keras.layers.Dense(self.num_classes, activation="softmax")

        self.optimizer = tf.keras.optimizer.Adam(learning_rate=0.01)

    def call(self, inputs):
        output = self.VGG16(inputs)
        output = self.dense1(output)
        return self.dense2(output)

    def loss(self, probabilities, labels):
        loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        return tf.reduce_mean(loss_function(labels, probabilities))

    def accuracy(self, probabilities, labels):
        #Checks if the class with highest probability is the same as the true class
        correct_predictions = tf.equal(tf.argmax(probabilities, 1), tf.argmax(labels, 1))
        #What portion of predictions are correct
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_images, train_dates):
    total_number_images_training = len(train_images)
    num_batches = total_number_images_training/model.batch_size
    #Note: not shuffling for now, might incorporate later

    #run one epoch
    for i in range(num_batches):
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

    num_batches = total_number_images_testing/model.batch_size

    for i in range(num_batches):
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
    print("test images shape is ", test_images.shape)
    test_dates = np.load('test_lab.npy')
    print("test date shape is ", test_dates.shape)
    train_images = np.load('train_img.npy')
    print("train images shape is ", train_images.shape)
    train_dates = np.load('train_lab.npy')
    print("train_dates shape is ", train_dates.shape)

    model = Model()

    num_epochs = 1 #NOTE: tune
    for epoch in range(num_epochs):
        train(model, train_images, train_dates) #NOTE: returns loss, can check values

    accuracy = test(model, test_images, test_dates)

    print("test accuracy is ", accuracy)
    #NOTE: can also look at train accuracy if we want to to help tune model

def get_batch(start_index, batch_size, images, dates):
    ending_index = start_index + batch_size
    return (images[start_index: ending_index], dates[start_index: ending_index])

# if __name__ == 'main':
#     main()

main()