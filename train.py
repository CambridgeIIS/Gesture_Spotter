import tensorflow as tf
from get_data import tfdata
import pickle
def train_classifier(model, Model, SAVE_PATH, EPOCHS,
                learning_rate, train_data, train_label, test_data, test_label, batch_size):
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=["sparse_categorical_accuracy"])
    model.summary()

    SAVE_PATH_model = SAVE_PATH + 'Model'

    trainData, trainsteps, \
    validData, validsteps,\
        = tfdata(train_data, train_label, test_data, test_label, batch_size)


    # Stop if the validation accuracy doesn't imporove for x epochs
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=2)

    # Reduce LR on Plateau
    reduceLR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=2)

    history_lstm = model.fit(trainData.repeat(),
                             steps_per_epoch=trainsteps,
                             validation_data=validData.repeat(),
                             validation_steps=validsteps,
                             epochs=EPOCHS,
                             # callbacks=[reduceLR,earlyStopping]
                             callbacks=[reduceLR]
                             )

    ## Save model
    model.save(SAVE_PATH_model)

    ## Save model
    with open(SAVE_PATH + "train_results.pickle".format(Model), "wb") as handle:
        pickle.dump(history_lstm.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved training history to res")
