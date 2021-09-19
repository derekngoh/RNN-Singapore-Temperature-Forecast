
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler 

import Utils

class RNN_Model():
    def __init__(self) -> None:
        self.scaler = None
        self.original_data = None
        self.original_test_data = None
        self.train_data = None
        self.test_data = None
        self.scaled_train_data = None
        self.scaled_test_data = None
        self.scaled_full_data = None
        self.train_gen = None
        self.test_gen = None
        self.full_data_gen = None
        self.main_model = None
        self.prediction = None

    def set_original_data(self, data):
        self.original_data = data

    def set_train_test_data(self, test_percent_split=10):
        train, test = Utils.train_test_split_by_index(self.original_data,test_percent_split)
        self.train_data = np.array(train)
        self.test_data = np.array(test)
        self.original_test_data = test

    def scale_train_test_data(self):
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.train_data)
        self.scaled_train_data = self.scaler.transform(self.train_data)
        self.scaled_test_data = self.scaler.transform(self.test_data)

    def scale_full_data(self):
        self.scaler = MinMaxScaler()
        self.scaled_full_data = self.scaler.fit_transform(self.original_data)

    def create_train_test_gen(self,length,batch_size, full=False):
        if (full):
            self.full_data_gen = TimeseriesGenerator(self.scaled_full_data, self.scaled_full_data,
                                length=length,
                                batch_size=batch_size)
            return
            
        self.train_gen = TimeseriesGenerator(self.scaled_train_data, self.scaled_train_data,
                            length=length,
                            batch_size=batch_size)
        self.test_gen = TimeseriesGenerator(self.scaled_test_data, self.scaled_test_data,
                            length=length,
                            batch_size=batch_size)
    
    def create_standard_model(self,length,features):
        main_model = Sequential()
        main_model.add(LSTM(150, activation='relu', input_shape=(length,features)))
        main_model.add(Dense(1))
        main_model.compile(optimizer='adam', loss='mse')
        self.main_model = main_model
        return self.main_model.summary()

    def fit_model_with_val(self,epochs,patience):
        early_stop = EarlyStopping(patience=patience)
        self.main_model.fit_generator(self.train_gen, epochs=epochs, callbacks=[early_stop], validation_data=self.test_gen)

    def fit_full_data(self,epochs):
        self.main_model.fit_generator(self.full_data_gen,epochs=epochs)

    def predict_data(self,feed_length,features,dataset,test_len):
        prediction = []

        first_batch = dataset[-feed_length:]
        current_batch = first_batch.reshape((1,feed_length,features))

        for i in range(test_len):
            current_pred = self.main_model.predict(current_batch)[0]
            prediction.append(current_pred)
            current_batch = np.append(current_batch[:,1:,:], [[current_pred]], axis=1)
        
        self.prediction = prediction

    def save_loss_plot(self,filename,show=False,figsize=(15,6)):
        Utils.save_show_loss_plot(self.main_model,filename,show=show,figsize=figsize)

    def set_current_path(self,filename=None):
        return Utils.get_set_current_path(filename)

    def save_text(self,content, filename):
        Utils.save_as_text_file(content,filename)
