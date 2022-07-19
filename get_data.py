
import numpy as np

import tensorflow as tf

from functools import partial

def tfdata(train_data, train_label, test_data, test_label, batch_size):


    trainData =  tf.data.Dataset.from_tensor_slices(
        (train_data, train_label)
    )
    trainData = trainData.shuffle(buffer_size=np.shape(train_data)[0])
    trainData = trainData.batch(batch_size).prefetch(buffer_size=1)
    trainsteps = np.shape(train_data)[0] // batch_size

    testData =  tf.data.Dataset.from_tensor_slices(
        (test_data, test_label)
    )
    testData = testData.shuffle(buffer_size=np.shape(test_data)[0])
    testData = testData.batch(batch_size).prefetch(buffer_size=1)
    teststeps = np.shape(test_data)[0] // batch_size

    return trainData,trainsteps, testData, teststeps


def getDataTest(test_data, test_label, batch_size):

    testData =  tf.data.Dataset.from_tensor_slices(
        (test_data, test_label)
    )

    testData = testData.batch(batch_size).prefetch(buffer_size=1)
    teststeps = np.shape(test_data)[0] // batch_size

    return testData, teststeps, test_data, test_label

