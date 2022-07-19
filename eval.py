import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score,
    f1_score
)
from get_data import getDataTest


def eval_classifier(Model,log,SAVE_PATH,test_data, test_label, batch_size):

    ## test load
    Data,_,_, y_true  = getDataTest(test_data, test_label, batch_size)
    SAVE_PATH_model = SAVE_PATH + 'Model'
    model = keras.models.load_model(SAVE_PATH_model, custom_objects={'tf': tf})
    log.info('Model loaded: {}'.format(Model))

    out = model.predict(Data, verbose=1)
    y_pred = np.argmax(out,axis=1)
    y_true = y_true.astype(np.int32)

    log.info('Real Data for validation')
    log.info('Accuracy: {:.4f}'.format(accuracy_score(y_pred, y_true)))
    log.info('Precision: {:.4f}'.format(precision_score(y_pred, y_true, average='macro')))
    log.info('Recall: {:.4f}'.format(recall_score(y_pred, y_true, average='macro')))
    log.info('F1 score: {:.4f}'.format(f1_score(y_pred, y_true, average='macro')))
    log.info('-------------------------------------------------------------------')
    log.info('\n')
    cm = confusion_matrix(y_pred, y_true)
    np.save(SAVE_PATH + '/confusion_matrix.npy', cm)

    return out,[accuracy_score(y_pred, y_true), precision_score(y_pred, y_true, average='macro'),
            recall_score(y_pred, y_true, average='macro'), f1_score(y_pred, y_true, average='macro') ]


