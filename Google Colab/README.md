Singapore Temperature "Forecast"


**Program and Goal**
The aim of this project was to "forecast" Singapore's annual average temperature of Year 2014 using historical data. The model was trained on a set of temperature data of Singapore, dating back to more than a 100 years. The model predicted an annual average of 27.26 degrees against the actual average of 27.9 degrees.


**The Data**
The data used in this program is the "Global Climate Change Data" made available by Data Society. This data can be downloaded from https://data.world/data-society/global-climate-change-data.


**Data Overview**
The data contains global information on monthly average temperature for every major city in the world since the 1800s. There exists some null values in the data, which were handled in the program. Besides the average monthly temperature and the corresponding month and year, the average temperature uncertainty, the city and country, latitude and longitutde are also contained in the data. 


**Structure & Approach**
The data relevant to Singapore was extracted and used as the dataset of focus. Null values were handled and dataset was preprocessed for model fitting. 10 years worth of data points (about 10%) were used as validation test sets while the rest were used to traing the model. Multiple fitting iterations were done to optimise the model parameters and configurations against the validation set. A final forecast was simulated for Year 2014 using a new model fitted to the entire dataset. Final simulated forecast temperature(mean) is checked with the actual 2014 annual average.

The program is divided into the following 6 sections:

Section 1 : Explore Data 
Section 2 : Data Preprocessing
Section 3 : Model Training and Fitting
Section 4 : Model Evaluation
Section 5 : Simulated "Forecast"
Section 6 : Results and Analysis 


**Result Analysis**
Model training was iterated with generator batch sizes between 12 to 48 points, LSTM nodes count between 100 to 200, hyperbolic tangent and relu activation function and early stopping patience of between 5 to 20. Performance was found to be good with a generator batch size of 24 (2 years), 150 nodes using relu activation and an early stopping patience of 10. 

- Validation Set RMSE = 0.466
- Forecasted Avg Temperature for 2014 (vs Actual) = 27.26 degrees (vs 27.9 degrees)
- Percentage Error of Forecast = 2.29%

With reference to the model with validation set, an epoch of 16 was found to be sufficient and hence used for the final "forecast" model. Trained model produced good results with 0.466 root mean square error when validated with the test set. Final model was trained using full dataset with parameters selected from the previous trained model. Final model accurately predicted average annual temperature of 2014, without prior knowledge of the data. The prediction fell within 2.29% of the actual data.
