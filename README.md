2014 Temperature "Forecast" (Singapore)

**Program and Goal**
The aim of this project was to "forecast" Singapore's annual average temperature of Year 2014 using historical data. The model was trained on a set of temperature data of Singapore, dating back to more than a 100 years. The model predicted an annual average of 27.26 degrees against the actual average of 27.9 degrees. 

**The Data**
The data used in this program is the "Global Climate Change Data" made available by Data Society. This data can be downloaded from https://data.world/data-society/global-climate-change-data.


**Data Overview**
The data contains global information on monthly average temperature for every major city in the world since the 1800s. There exists some null values in the data, which were handled in the program. Besides the average monthly temperature and the corresponding month and year, the average temperature uncertainty, the city and country, latitude and longitutde are also contained in the data. 


**How to Run**
Download "data" folder
Download the following 4 Python files:
1. Temperature Trainer.py
2. Temperature Forecast.py
3. RNN_Model.py
4. Utils.py

To run the training model and see the RMSE of the validation set, do the following.
1. In the terminal, cd into the directory where the downloaded files are. Make sure files are organised as shown.
2. Enter command "Temperature_Trainer.py [-h] [--test_percent TEST_PERCENT] [--gen_feed GEN_FEED] [--training_epochs TRAINING_EPOCHS]"
3. Example: python Temperature_Trainer.py --test_percent 20 --gen_feed 12 --training_epochs 30

To train the model with the full dataset and "forecast" the annual temperature of 2014, do the following.
1. In the terminal, cd into the directory where the downloaded files are. Make sure files are organised as shown.
2. Enter command "Temperature_Forecast.py [-h] [--forecast_length FORECAST_LENGTH] [--gen_feed GEN_FEED] [--training_epochs TRAINING_EPOCHS]"
3. Example: python Temperature_Forecast.py --forecast_length 30 --gen_feed 12 --training_epochs 10


**Result**
Model training was iterated with generator batch sizes between 12 to 48 points, LSTM nodes count between 100 to 200, hyperbolic tangent and relu activation function and early stopping patience of between 5 to 20. Performance was found to be good with a generator batch size of 24 (2 years), 150 nodes using relu activation and an early stopping patience of 10. 

- Validation Set RMSE = 0.466
- Forecasted Avg Temperature for 2014 (vs Actual) = 27.26 degrees (vs 27.9 degrees)
- Percentage Error of Forecast = 2.29%

With reference to the model with validation set, an epoch of 16 was found to be sufficient and hence used for the final "forecast" model. Trained model produced good results with 0.466 root mean square error when validated with the test set. Final model was trained using full dataset with parameters selected from the previous trained model. Final model accurately predicted average annual temperature of 2014, without prior knowledge of the data. The prediction fell within 2.29% of the actual data.

**Requirements**
1. Python 3.5 or higher
2. Tensorflow version 2.6.0 or higher