import os, argparse, random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from RNN_Model import RNN_Model

cmdparser = argparse.ArgumentParser(description='sg temperature forecast 2014')
cmdparser.add_argument('--forecast_length', help='set how many months to be forecasted', default='15')
cmdparser.add_argument('--gen_feed', help='set length of feed of generator', default='24')
cmdparser.add_argument('--training_epochs', help='number of epochs to train for', default='2')
args = cmdparser.parse_args()

arg_forecast_length = int(args.forecast_length)
arg_gen_feed = int(args.gen_feed)
arg_train_epochs = int(args.training_epochs)


#LOAD and PREPROCESS data
current_file_loc = os.path.dirname(os.path.realpath(__file__))
file_loc = os.path.join(current_file_loc, "data\SingaporeTemperatureData.csv")
raw_df = pd.read_csv(file_loc)

df = raw_df.drop(['City','Country','Latitude','Longitude'], axis=1) 
date1 = pd.to_datetime(df['dt'], format='%Y-%m-%d', errors='coerce') 
date2 = pd.to_datetime(df['dt'], dayfirst='%d/%m/%y', errors='coerce')
df['dt'] = date1.combine_first(date2)
df['Year'] = df['dt'].dt.year
df['Month'] = df['dt'].dt.month
df['Century'] = df['Year'].apply(lambda year: year//100)
df['AverageTemperature'] = df.groupby('Year')['AverageTemperature'].transform(lambda x: x.fillna(x.mean()))
df['AverageTemperature'] = df.groupby('Century')['AverageTemperature'].transform(lambda x: x.fillna(x.mean()))
df = df[df['Year']>1899] 
df = df.drop(['Year', 'Month','Century','AverageTemperatureUncertainty'], axis=1)
df = df.set_index('dt')

random_num = random.randint(0,999)

#CREATE new model and 'forecast' into 2014
length_of_fgen_feed = arg_gen_feed
fgen_batch_size = 1
fgen_features = 1
fgen_epochs = arg_train_epochs #15 is the value obtained from previous training model in Google Colab
forecast_len = arg_forecast_length #number of points/ months to forecast into.

forecast_model = RNN_Model()
forecast_model.set_original_data(df)
forecast_model.scale_full_data()
forecast_model.create_train_test_gen(length_of_fgen_feed,fgen_batch_size,full=True)
forecast_model.create_standard_model(length_of_fgen_feed,fgen_features)
forecast_model.fit_full_data(fgen_epochs)
forecast_model.main_model.save(forecast_model.set_current_path("temp_forecast_model{a}.h5".format(a=random_num)))
forecast_model.predict_data(length_of_fgen_feed,fgen_features,forecast_model.scaled_full_data,forecast_len)
forecastdf = forecast_model.scaler.inverse_transform(forecast_model.prediction)

#As we are going to plot an extension into the dataframe, we create here an extension of the indexes to be used.
index_for_forecast = pd.date_range('2013-10-01', periods=forecast_len, freq='MS')

#Add the "forecasts" to the existing data. 
df_fc = pd.DataFrame(data=forecastdf,index=index_for_forecast,columns=['Avg Temp 2014'])

#View the existing data and the "forecast" in a plot
axis = df.plot()
df_fc.plot(ax=axis, figsize=(16,8))
plt.savefig(forecast_model.set_current_path("forecast_plot{a}.jpg".format(a=random_num)))
plt.xlim('2010-01-01','2014-12-01')
plt.show()

#Get the average forecasted temperature of 2014
Forecasted_2014_AnnualAvg = df_fc['2014-01-01':'2014-12-01'].mean()[0]

#This is the actual average temperature for 2014 retrieved from https://www.weather.gov.sg/climate-past-climate-trends/
Actual_2014_AnnualAvg = 27.9

#Check percentage error of the "forecast"
percentage_error = (100*(Forecasted_2014_AnnualAvg-Actual_2014_AnnualAvg)/Actual_2014_AnnualAvg)

result_text = "\nThe forecasted 2014 Annual Average is {a: .2f} while the actual annual average is {b: .2f}".format(a=Forecasted_2014_AnnualAvg,b=Actual_2014_AnnualAvg)
result_text2 = "\n\nThe percentage error is {c: .2f}%".format(c=percentage_error)
forecast_model.save_text(result_text+result_text2,"forecasted_result{a}.txt".format(a=random_num))
print(result_text,result_text2)