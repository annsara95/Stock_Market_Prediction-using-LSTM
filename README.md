# Stock_Market_Prediction-using-LSTM
Final Project on CSYE 7245

Predicting stock Market prices gets very difficult as it is highly unpredictable. Hence, forecasting stock prices gets very tricky. Traditional forecasting algorithms like ARIMA, SARIMA, Moving Average methods dont give out the best predictions. In this project, we will be looking into using RNN (Recurrent Neural Network)'s model - LSTM (Long-Short-term memory) which uses feedback connections along with forwardfeebacks.
LSTMs were developed to deal with the vanishing gradient problem that can be encountered when training traditional RNNs.

<b>LSTM with Open prices and Previous day Open prices:</b>

In this method, along with the Open prices we also pass the previous day open prices. LSTM model's predictions are better in this case as it gets the previous day open prices as well in input.

<b>LSTM with Open prices and twitter sentiments measures:</b>

We will be also be looking into see how Twitter sentiments affects the forecasting. We have a methodology that will feed stock prices as well as a sentiment measures to LSTM for forecasting. 
![Methodology](https://github.com/annsara95/Stock_Market_Prediction-using-LSTM/blob/master/BDIA%20Project/LSTM_Method.png)

<b> Conclusion:</b>

Though the LSTM model with twitter sentiments results were not good, it was an attempt to see how the sentiments affect the predictions. Using tweets from specific days would not affect directly, we will need more external information on prediction results.

