# ## 1 Вариант
# недоделанный

from tensorflow import keras
from keras import layers, backend
import numpy as np
from ReLU import ReLU_new
from Conv import Conv
from Pool import Pool
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, ReLU, MaxPooling2D, Flatten, Dense
from keras.callbacks import Callback
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mlxtend.preprocessing import minmax_scaling

def trainConv(X, D, hidden_layers, CC, epoch):
    # Сброс глобального состояния Keras
    backend.clear_session()
    model = Sequential()

    a = 10
    WC = -a * np.ones(CC)

    i = 1
    OldPerformance = 10

    class CBNNTrainingCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(f"Epoch {epoch} started")

    callback = CBNNTrainingCallback()

    while i <= epoch:
        print('\n>>> Epoch ', i, '/', epoch)
        callback.on_epoch_end(i)

        step = 2 * a / epoch
        WC += step

        x_flattened = []
        for k in range(len(D)):
            x = X[:, :, k]
            yC1 = Conv(x, WC)
            yC2 = ReLU_new(yC1)
            yC = Pool(yC2)

            yC_f = np.reshape(yC, -1)
            x_f = np.reshape(x, -1)
            x_flattened.append(np.concatenate((yC_f, x_f)))

        x = np.array(x_flattened)
        t = D

        # print('\n\tx.shape = ', x.shape)
        # print('\tx = ', x)
        # print('\tx[0] = ', x[0])
        # print('\n\tt.shape = ', t.shape)
        # print('\tt = ', t)

        # ================================================================================
        # Новая модель обучения

        # Split the data
        x_train = x
        y_train = t
        # y_train = minmax_scaling(y_train, columns=[0]) # scaling between 0 and 1
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15, shuffle=True)
        print('\n\ttrain.shape / val.shape = ', x_train.shape[0], x_val.shape[0])
        # print('\n\tx_train.shape = ', x_train.shape)
        # print('\tx_train = ', x_train)
        # print('\n\tx_val.shape = ', x_val.shape)
        # print('\tx_val = ', x_val)
        # print('\n\ty_train.shape = ', y_train.shape)
        # print('\ty_train = ', y_train)
        # print('\n\ty_val.shape = ', y_val.shape)
        # print('\ty_val = ', y_val)

        # Preprocess the data (NumPy arrays):
        # x_train = x_train.reshape(7604, 105).astype("float32")
        # y_train = y_train.astype("float32")

        # Model build
        inputs = keras.Input(shape=(None, 6083, 105))  # shape=x.shape[1:]
        x = layers.Dense(105, activation="linear", name="dense_1")(inputs)
        # x = layers.Dense(105, activation="relu", name="dense_2")(x)
        x = layers.Dense(210, activation="softplus", name="dense_3")(x)
        x = layers.Dropout(0.3, name="dropout_1")(x)
        x = layers.Dense(210, activation="softplus", name="dense_4")(x)
        x = layers.Dense(105, activation="softplus", name="dense_5")(x)
        # x = layers.Dropout(0.3, name="dropout_2")(x)
        x = layers.Dense(105, activation="softplus", name="dense_6")(x)
        # x = layers.Dense(105, activation="softmax", name="dense_7")(x)
        outputs = layers.Dense(1, activation="linear", name="predictions")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        callbacks = [
            # Остановить обучение если `val_loss` перестанет улучшаться в течение [patience] эпох
            keras.callbacks.EarlyStopping(patience=10, monitor='val_loss'),
            # Записать логи TensorBoard в каталог `./logs` directory
            # keras.callbacks.TensorBoard(log_dir='./logs')
        ]

        model.compile(
            optimizer=keras.optimizers.Adam(0.001),  # Optimizer
            # Minimize loss:
            loss='mse',
            # Monitor metrics:
            metrics=['mae'],
        )

        # Model training
        print(">>> Fitting model on training data")
        history = model.fit(
            x_train,
            y_train,
            batch_size=105,
            epochs=100,
            # Validation of loss and metrics
            # at the end of each epoch:
            validation_data=(x_val, y_val),
            callbacks=callbacks
        )

        # Evaluating model
        print("\n>>> Evaluating model on test data")
        results = model.evaluate(x_val, y_val, batch_size=105)
        print("\tEvaluation results: ", results)

        # Generate a prediction using model.predict()
        # and calculate it's shape:
        print("\n>>> Generating a prediction")
        prediction = model.predict(np.array(x_flattened))
        print("\tprediction shape:", prediction.shape)
        print("\tprediction =", prediction)
        y = prediction



        # ================================================================================
        # Графики и расчет производительности

        print('\tt.shape = ', t.shape)
        print('\ty.shape = ', y.shape)
        performance = np.mean((t - y) ** 2)
        pred_error = [(a - b) for a, b in zip(t, y)]
        # print('\tperformance = ', performance)
        # print('\tt = ', t)
        # print('\ty.shape = ', y.shape)
        # print('\ty = ', y)
        # count = 0
        # for p in range(y.shape[1]):
        #     if y[0][p] != 0:
        #         count += 1
        # print('\tNot-zero number of \'y\'= ', count)

        plt.figure(figsize=(10, 10))
        plt.subplot(211)
        plt.plot(t, 'b.', label='D_max')
        plt.plot(prediction, 'r.', label='Prediction')
        plt.legend(loc="upper left")
        plt.subplot(223)
        plt.plot(t, prediction, 'b.')
        plt.axline((1, 1), slope=1, ls='--', c='r')
        plt.xlabel('D_max')
        plt.ylabel('Predicted D_max')
        plt.xlim(0, 80)
        plt.ylim(0, 80)
        plt.subplot(224)
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.plot(history.history['loss'], label='loss')
        plt.ylim(min(history.history['val_loss'])-5, max(history.history['val_loss'])+5)
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.legend(loc="upper right")
        plt.suptitle('Prediction %i' % i)
        plt.show()

        if performance < OldPerformance:
            print('\n>>> !!! Performance is BETTER than the previous ones > result can be passed')
            WC_new = WC
            NET = model
            TR = history
            best_prediction = prediction
            best_prediction_epoch = i
            best_pred_error = pred_error
            best_history = history

            OldPerformance = performance

        elif i == 1:
            print('\n>>> !!! Performance is BETTER than the previous ones > result can be passed')
            WC_new = WC
            NET = model
            TR = history
            best_prediction = prediction
            best_prediction_epoch = i
            best_pred_error = pred_error
            best_history = history

            OldPerformance = performance

        else:
            print('\n>>> Performance is WORSE than the previous ones > result cannot be passed')

        i += 1

    plt.figure(figsize=(10, 10))
    plt.subplot(211)
    plt.plot(t, 'b.', label='D_max')
    plt.plot(best_prediction, 'r.', label='Best prediction')
    plt.legend(loc="upper left")
    plt.subplot(223)
    plt.plot(t, best_prediction, 'b.')
    plt.axline((1, 1), slope=1)
    plt.xlabel('D_max')
    plt.ylabel('Predicted D_max')
    plt.xlim(0, 80)
    plt.ylim(0, 80)
    plt.subplot(224)
    plt.plot(best_history.history['val_loss'], label='val_loss')
    plt.plot(best_history.history['loss'], label='loss')
    plt.ylim(min(best_history.history['val_loss'])-5, max(best_history.history['val_loss'])+5)
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend(loc="upper right")
    plt.suptitle('Best prediction is %i' % best_prediction_epoch)
    plt.show()

    return WC_new, NET, TR
