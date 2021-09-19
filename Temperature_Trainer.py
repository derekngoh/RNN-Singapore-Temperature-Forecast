import os, argparse, random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from RNN_Model import RNN_Model

cmdparser = argparse.ArgumentParser(description='sg temperature trainer 2014')
cmdparser.add_argument('--test_percent', help='select proportion for test set', default='10')
cmdparser.add_argument('--gen_feed', help='set length of feed of generator', default='24')
cmdparser.add_argument('--training_epochs', help='number of epochs to train for', default='30')
args = cmdparser.parse_args()

arg_test_percent = int(args.test_percent)
arg_gen_feed = int(args.gen_feed)
arg_train_epochs = int(args.training_epochs)


#LOAD data
current_file_loc = os.path.dirname(os.path.realpath(__file__))
file_loc = os.path.join(current_file_loc, "data\SingaporeTemperatureData.csv")
raw_df = pd.read_csv(file_loc)

#PREPROCESS data with prior knowledge after iterations done on Google Colab
df = raw_df.drop(['City','Country','Latitude','Longitude'], axis=1) 
date1 = pd.to_datetime(df['dt'], format='%Y-%m-%d', errors='coerce') 
date2 = pd.to_datetime(df['dt'], dayfirst='%d/%m/%y', errors='coerce')
df['dt'] = date1.combine_first(date2)

df['Year'] = df['dt'].dt.year
df['Month'] = df['dt'].dt.month
df['Century'] = df['Year'].apply(lambda year: year//100)

df['AverageTemperature'] = df.groupby('Year')['AverageTemperature'].transform(lambda x: x.fillna(x.mean()))
df['AverageTemperature'] = df.groupby('Century')['AverageTemperature'].transform(lambda x: x.fillna(x.mean()))

df = df[df['Year']>1899] #drop earlier values as iterations show it produces better results.
df = df.drop(['Year','Century','Month','AverageTemperatureUncertainty'], axis=1) #drop columns which are no longer useful
df = df.set_index('dt')

#TRAIN-TEST GENERATOR. Create train and validation test sets and their generators.
test_percent_split = arg_test_percent
length_of_generator_feed = arg_gen_feed
batch_size = 1

valtest_model = RNN_Model()
valtest_model.set_original_data(df)
valtest_model.set_train_test_data(test_percent_split)
valtest_model.scale_train_test_data()
valtest_model.create_train_test_gen(length_of_generator_feed,batch_size)

random_num = random.randint(1,999)

#FIT model. Number of nodes and activation function selected after iterations on Google Colab.
num_of_data_features = 1
early_stop_patience = 5
epochs = arg_train_epochs

valtest_model.create_standard_model(length_of_generator_feed,num_of_data_features)
valtest_model.fit_model_with_val(epochs,early_stop_patience)
valtest_model.main_model.save(valtest_model.set_current_path("temp_test_model{a}.h5".format(a=random_num)))
valtest_model.save_loss_plot("training_loss_plot{a}.jpg".format(a=random_num), show=True)

#VALIDATE data with test set
valtest_model.predict_data(length_of_generator_feed,num_of_data_features,valtest_model.scaled_test_data,len(valtest_model.scaled_test_data))
valtest_model.original_test_data['Validation Set Prediction'] = valtest_model.scaler.inverse_transform(valtest_model.prediction)
valtest_model.original_test_data.plot(figsize=(15,6))
plt.savefig(valtest_model.set_current_path("prediction_plot{a}.jpg".format(a=random_num)))

#PERFORMANCE check on validation set using RMSE
rmse = np.sqrt(mean_squared_error(valtest_model.original_test_data['AverageTemperature'],valtest_model.original_test_data['Validation Set Prediction']))
rmse_text = "Root Mean Square Error of Validation Set Prediciton: " + str(np.round_(rmse,2))
valtest_model.save_text(rmse_text,"rmse_result{a}.txt".format(a=random_num))
print(rmse_text,"\n\n")
