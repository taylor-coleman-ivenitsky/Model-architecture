class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        self.batch_size = 100
        self.num_classes = ?
        self.loss_list = []

        self.optimizer = tf.keras.optimizer.Adam(learning_rate=1e)
