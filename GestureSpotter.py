from functools import partial
import numpy as np
import editdistance
import time

# Reduce tensorflow verbosity
import os
# Cut back on TF output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Disable CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
from tensorflow import keras


maxp = 0.31
minp= 0.05

class GestureSpotter:
    def __init__(self, model_path,input_dim):
        self.recur = 0
        self.prev_spot_class = -1
        self.model = keras.models.load_model(model_path, custom_objects={'tf': tf})     
        print('Model loaded from: {}'.format(model_path))
        print(tf.config.list_physical_devices('GPU'))        
        # Run a dummy predict, as Keras otherwise slow on first call when using CUDA
        dummy_data_frame = np.zeros((1,20,input_dim))
        self.predict(dummy_data_frame)

    def predict(self, data_frame):
        Data = tf.data.Dataset.from_tensor_slices(
            (data_frame)
        )
        Data = Data.batch(1).prefetch(buffer_size=1)

        outputs = self.model.predict(Data, verbose=0)
        outputs_softmax = tf.nn.softmax(outputs)

        # Normalize
        outputs_normalized = outputs_softmax.numpy()        
        for i in range(len(outputs_normalized)):
            # TODO want to avoid these magic numbers, or at least configure as class params
            outputs_normalized[i] = (outputs_normalized[i] -  minp) / (maxp - minp)

        y_pred = np.argmax(outputs_normalized, axis=1)
        y_pred_value = np.max(outputs_normalized)
        return (y_pred, y_pred_value, outputs_normalized)

    def spot(self, data_frame, detect_threshold, recur_threshold):
        # set spot_class to -1, indicating no detection
        spot_class = -1

        # predict on current frame
        # t_start = time.time()
        prediction = self.predict(data_frame)
        # print('Predict time: {} ms'.format( (time.time() - t_start) * 1000))

        gesture_class = prediction[0]
        gesture_class_value = prediction[1]
        gesture_probabilities = prediction[2]




        if gesture_class_value > detect_threshold:
            self.recur += 1
            if self.recur > recur_threshold:
                self.recur = 0
                if gesture_class != self.prev_spot_class:
                    spot_class = gesture_class
                    self.prev_spot_class = spot_class

        # if gesture_class_value > detect_threshold:
        #     if gesture_class != self.prev_spot_class:
        #
        #         self.prev_spot_class = spot_class
        #         self.recur = 0
        #     else:
        #         self.recur += 1
        #         if self.recur == recur_threshold:
        #             spot_class = gesture_class
        #             self.recur = 0

        else:
            spot_class = 0
            
        return (spot_class, gesture_probabilities)


