# -Caching-a-PyTorch-Deep-Neural-Network

In this analysis, stocks were filtered based on financial criteria, including a requirement for a launch date exceeding one year and trailing twelve months (TTM) earnings greater than 0. The data was then partitioned into training and testing sets. Input features for a momentum model were derived from the highest price, lowest price, opening price, and closing price over the past four days, with momentum synthesized in various forms. A closely linked fully-connected neural network was constructed to model momentum, particularly associated with a three-day yield. The testing sample's rates of return were computed, incorporating profitability measures and the Sharpe ratio.

I develop the code in a quantitive trading platform(https://bigquant.com/), which provides reliable data and backtest system

Also, I provide a similiar reinforcement learning trading in China A-stock based on A2C, the neural network underlying the A2C algorithm is a structure composed entirely of linear layers. Structurally, it appears similar to a Deep Neural Network (DNN), with the key distinction lying in the optimization approach. While DNNs aim to minimize error, A2C focuses on maximizing the cumulative reward. This difference in optimization strategies is the primary contrast between the two, despite their comparable structural resemblance. 
