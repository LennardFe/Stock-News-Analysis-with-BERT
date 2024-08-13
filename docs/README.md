![th_leeway_small](https://github.com/user-attachments/assets/042c0661-fc27-4eca-9e5e-2d39255b87bd)

<div align="justify"><b>Note:</b> The following work was developed as part of a project at the Cologne University of Applied Sciences in cooperation with the start-up company Leeway. The project took place over a period of ten months and includes both the so-called practical project and the code of the typical bachelor thesis. The practical project focuses on the practical implementation of the objectives defined in the following text and was specifically designed and developed for this purpose. The main elements of the practical project can be found in the file <i>nb_practicalproject.ipynb</i>. The bachelor thesis adds a scientific aspect to this project. The associated code, which extends the previous practical project, is contained in the file <i>nb_bachelorthesis.ipynb</i>. Please note that this repository contains only the code intended for public use. Some parts of the code are missing, indicated with comments, as they are not intended for public distribution. The complete repository will remain private.</div>

---

# <div align="justify">Practical-Project in WS 23/24: Developing a BERT-based Language Model with Advanced Data Processing for Stock News Analysis</div>

Supervisors: __Prof. Dr. Johann Schaible and Lars Wißler__\
Elaboration by: __Lennard Feuerbach__

<div align="justify">This project aims to analyze the relationship between stock news and following price movements using a BERT-based language model. Unlike traditional sentiment analysis, my approach involves direct prediction of future stock changes based on the content of news articles. I cover large steps to process the data, ensuring a usable and consistent format to feed into and to test my model. The data is retrieved through the <a href="https://leeway.tech/data-api">Leeway</a> API, which provides me with all the necessary data I need.</div>

<br>

## Table of Contents

1. [Introduction](#introduction)
2. [Methodology](#methodology)
3. [Setup & Configuration](#setup--configuration)
4. [Examples](#examples)
5. [Results & Interpretation](#results--interpretation)
6. [Challenges & Limitation](#challenges--limitation)
7. [Conclusion](#conclusion)

<br>

## Introduction

### Motivation
<div align="justify">
<p>The uprise of Large Language Models such a BERT, Llama, GPT and many others, has revolutionized text analysis, particularly the field of sentiment analysis. These models have found huge success accross different sectors, including in the world of finance, where they have already been utilized for various tasks to extract valuable insights from textual data.</p>
</div>

<div align="justify">
<p>In this project, I bypass the conventional sentiment analysis step and directly connect news articles to their following corresponding  stock movements. This method offers me a clear advantage in the training phase, eliminating the time consuming need to manually label the training data. Alongside my simplified and more efficient training phase, I also provide the capability to directly observe and understand the varying impact of each news on stock movements. This enables me to adjust more effectively and to better understand the fluctuations of the stock market.</p>
</div>

### Background
<div align="justify">
<p>The first publications in the context of sentiment analysis, natural language processing, and text processing were published nearly 20 years ago¹, obviously in a much smaller state than we are in right now. A milestone was the 2017-published paper "Attention is All You Need" by Google Research², first introducing the transformer architecture, which is used in the 2018-released BERT-Model. Since then, there has been a consistent upward trend, with different GPT versions, LLama, and many others. Following the release of ChatGPT, the concept of these models has gained general acceptance and usage, even beyond the tech sphere. With the rise of this technology, other sectors also started looking into how they can incorporate machine learning in general, including text processing and sentiment analysis.</p>
</div>

<div align="justify">
<p>The finance sector is no different, a big and aggressive competition where everyone is trying to be ahead and to be the first to gain deeper insights and more knowledge about the situation of the financial markets. There is a growing trend towards automation driven by artificial intelligence. Trading strategies, supported by various algorithms in the backend, are becoming increasingly widespread. Both private traders and large firms are relying more on these techniques, aiming to adapt to market changes faster than their counterparts. Working with the stock market and attempting to predict or calculate the influence of news or other inputs on stock prices is obviously no easy task. Market dynamics are influenced by a lot of different factors, such as geopolitical events, economic indicators, and unpredictable circumstances. Nonetheless, this project aims to find out, if there is a significant enough correlation between the content of news articles and their following price changes. </p>
</div>

### Significance
<div align="justify">
<p>As mentioned earlier, the project directly predicts stock price movement based on the content of news articles. The model is trained on the news content with a corresponding label containing the price change of the stock, from release date of the news, to the given target date. This even helps me, to better understand each decision of the model, since I can backtrack the prediction to the corresponding news. Because I skip the step of calculating sentiment values and interpreting them, the model becomes more versatile in its potential use cases. This enables faster or even real-time decision-making by reducing in-between steps, resulting in a possible competitive advantage for the user.  Additionally the training phase is shortened, since the need for manual labeling is not necessary anymore. Because no human has to label the data, I completely removed the risk of labeling bias, which is the chance of prejudice in the labeled data.</p>
</div>

<br>

## Methodology

<div align="justify">
<p>In my linked <a href="https://github.com/LennardFe/Stock-News-Analysis-with-BERT/blob/main/docs/methodology.md">methodology</a>, I examine the entire process from raw data to the final valuable insights. Additionally, I will explore the techniques and methods applied, along with the reasons for my choices. Throughout the explanation, I'll include code snippets to support understanding. For more and complete information regarding the functions, refer to the complete docstrings in the notebook.<p>
</div>

<br>

## Setup & Configuration

<div align="justify">
<p>The <a href="https://github.com/LennardFe/Stock-News-Analysis-with-BERT/blob/main/docs/setup.md">setup</a> provides essential instructions for initiate the environment and configuring the project effectively. It offers step-by-step guidance to prevent mistakes and includes troubleshooting tips that I've encountered while setting up the environment myself. Please stick to the recommended versions of Python and packages, as any variation could result in errors in the code.<p>
</div>

<br>

## Examples

### Two-Week Stock Change Prediction
<div align="justify"> 
<p>In this example, I'll showcase the usage of the code for a complete execution from start to end. The provided snippets are extracted from the original notebook, focusing on predictions generated by my model for stock price movements in <b>two weeks</b>. The full version of this example is available here: <a href="https://github.com/LennardFe/Stock-News-Analysis-with-BERT/blob/main/results/notebook_2W_FULL_PERCENTAGE_10.html">notebook_2W_FULL.html</a>, containing the executed code, coherent functions, extra comments, and docstrings. Additionally, the most important parameters and function calls are explained more throughout in the <a href="https://github.com/LennardFe/Stock-News-Analysis-with-BERT/blob/main/docs/methodology.md">methodology</a>.</p> 
</div>

<div align="justify"> 
<p>I begin by training a DistilBERT model to analyze news articles related to the stocks of the ETFs "XLP.US" and "XLK.US". I cover a time period from January 2021 to the end of 2022, utilizing the New York Stock Exchange trading day calendar, to determine the opening hours and holidays. As previously mentioned, I work with a two-week target. This means there is a two-week gap between the release date of the news and the stock price change prediction I provide. The dataset is divided into a training, validation and a testing set. To achieve equal representation in each new dataset, the data is grouped by ticker and then distributed accordingly. For instance, for all AAPL news, 70% are allocated to the training set, 15% to the validation set, and the remaining 15% to the testing set. This distribution rule applies to all other news as well. All data from the then-created training subset is taken since we don't specify a specific <i>sample_size</i>. The <i>thresh_mode</i> is used to determine the mode for calculating the threshold. The choosen percentage mode, considers the top 10% for BUY and bottom 10% for the SELL signals. Lastly, the top and worst 100 predicted news, as specified in the <i>num_pred</i> argument, are displayed alongside with the overall profit or loss. The results of this example are fully covered in the <a href="https://github.com/LennardFe/Stock-News-Analysis-with-BERT/blob/main/docs/methodology.md">Results and Interpretation</a> section right below.</p>
</div>

**Code:**
```python
# Configuration variables for the data to use
etfs            = ["XLP.US", "XLK.US"]      # List of ETFs to be used in the analysis
start_date      = "2021-01-01"              # Start date of the interval
end_date        = "2022-12-31"              # End date of the interval
calendar        = StockExchange.NYSE        # Stock exchange calendar to be used                
target          = Target.W2                 # Target to predict on                            


# Configuration variables for the BERT model to train or load if model_name already exists
learning_rate   = 1e-5                      # Learning rate for the model
epochs          = 3                         # Number of epochs to train the model
batch_size      = 16                        # Batch size for the model
weight_decay    = 0.015                     # Weight decay for the model        
model_type      = "distilbert-base-uncased" # Type of the model to be used                      
model_name      = f"DBERT_{target}_V1"      # Name of the model to be trained / loaded          


# Configuration variables for the simulation / evaluation
sample_size     = None                      # Number of samples from the dataset, None = all rows
random_seed     = 43                        # Random seed for sample size and reproducibility
thresh_mode     = TMode.PERCENTAGE          # Threshold calculation mode to be used             
threshold       = 10                        # Threshold value to be used                        
num_predictions = 100                       # Amount of best/worst predictions to be displayed
```
```python
# data retrieving
stocks_df	= extract_from_etf(etfs) # stocks from etfs
dates_js	= get_trading_days(start_date, end_date, calendar) # trading days in time frame
values_js	= get_values(stocks_df, start_date, end_date) # values for stocks in time frame
news_df		= get_content(stocks_df, start_date, end_date) # news for stocks in time frame

# data processing
up_news_df	= update_dates(news_df, dates_js, target) # update dates to valid in time frame 
changes_df	= calc_changes(up_news_df, values_js, target) # calc changes from news to target
prep_df		= preprocess(changes_df) # preprocess the content

train, test, val	= split_df_by_ticker(prep_df) # split to keep representative representation
text, label, label_date	= get_label_for_target(target) # get column labels 

test	= adjust(test, text, label, label_date) # rename columns
train	= adjust(train, text, label) # rename and drop some columns
val	= adjust(val, text, label) # rename and drop some columns

# training / loading the language model (bert)
model	= bert(model_name, train, val, model_type, learning_rate, epochs, batch_size, weight_decay)

# simulating
s_test	= sample_rows(test, sample_size, random_seed) # reduce test size
sim	= simulate_predictions(model, s_test, mode, threshold) # simulate predictions and signals

# evaluating
best, worst, combined_df = evaluate_performance(sim, target, num_pred) # compare pred. to actual
```


### More examples

- Three-Day Stock Change Prediction: <a href="https://github.com/LennardFe/Stock-News-Analysis-with-BERT/blob/main/results/notebook_3D_FULL_PERCENTAGE_10.html">notebook_3D_FULL.html</a>
- One-Week Stock Change Prediction: <a href="https://github.com/LennardFe/Stock-News-Analysis-with-BERT/blob/main/results/notebook_1W_FULL_PERCENTAGE_10.html">notebook_1W_FULL.html</a>

<br>

## Results & Interpretation

<div align="justify">
<p>The following results, consisting of tables, graphs, and numerical data, come directly from the two-week example shown above. In this section, we dive into the interpretation of these results to extract insights and significance of the analysis.</p>
</div>

<br>

<table>
  <thead>
    <tr>
      <th width="550px">Epoch</th>
      <th width="550px">Training Loss</th>
      <th width="550px">Validation Loss</th>
      <th width="550px">Rmse</th>
      <th width="550px">Std</th>
      <th width="550px">(Rmse-Std)/Std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">1</td>
      <td align="center">25.698900</td>
      <td align="center">25.644241</td>
      <td align="center">5.064014</td>
      <td align="center">5.070770</td>
      <td align="center">-0.001333</td>
    </tr>
    <tr>
      <td align="center">2</td>
      <td align="center">25.506100</td>
      <td align="center">25.493818</td>
      <td align="center">5.049140</td>
      <td align="center">5.070770</td>
      <td align="center">-0.004266</td>
    </tr>
    <tr>
      <td align="center">3</td>
      <td align="center">24.352200</td>
      <td align="center">25.824621</td>
      <td align="center">5.081793</td>
      <td align="center">5.070770</td>
      <td align="center">0.002174</td>
    </tr>
  </tbody>
</table>

<div align="justify">
<p>To begin, I fine-tune (or load, if already available) a DistilBERT base model, which is smaller in size yet retains much of the performance and accuracy of the original BERT model. I've trained all my models for only three epochs, due to my limited resources and time. In the <a href="https://github.com/LennardFe/Stock-News-Analysis-with-BERT?tab=readme-ov-file#future-work">future</a>, this can and should be trained longer and with more focus on tuning parameters, to squeeze the best possible model out of our data. Next to the well-known training and validation loss, I also present the root mean square error (RMSE) and standard deviation (STD). These metrics are utilized to evaluate the model and to make it comparable with other trained models across different target dates. As I've worked with smaller target windows, such as three days or one week, the loss for these models will naturally be lower. However, this doesn't automatically mean that models trained on smaller targets are better. Larger target windows naturally have bigger fluctuations. To get a quick overview, about the accuracy of our model, I have to set the loss and corresponding RMSE in correlation to the standard deviation. This approach allows for comparing different targets with each other.
</p>
</div>

<br>

![302985552-b64b8b01-4acd-43e3-88d5-7e26a1973054](https://github.com/user-attachments/assets/c187940c-366a-4b5b-a874-137592c39f69)

<div align="justify">
<p>I calculate the threshold based on the predicted changes, made by our model. Since I specified the mode as "PERCENTAGE" and set it to 10%, my model sets the BUY signal for all predictions on news that are in the top 10% overall. The same goes for the worst 10%, which receive a SELL signal. All others in between stick to the HOLD signal. The two graphs give a good overview of the distribution of our predicted and the actual changes. As we can see, the actual change is near to a normal distribution, whereas our predicted ones are a bit right-skewed. It's important to mention that the actual change is the change of each stock, adjusted to the change of the related ETF. The positive threshold has been set to a change larger than 1.28%, the negative threshold to smaller than -0.77%. Below these two graphs, we can see that the mean value for both are close together, the same can't be said for the standard deviation. The difference can be explained through the fact that there are some outliers in the actual changes, with movements up to +40% / -25%, which results in a much wider spread standard deviation. While our model, even though it's trained on a comparable dataset, won't predict up to nearly 40% change in two weeks, since it has learned that this is quite unlikely. Theoretically, I could adjust some risk parameters to make the model more risk-friendly, which would result in a larger standard deviation for the predicted changes.</p>
</div>

<br>

```
Calculations for a target in 2W
Adjusted Change for BUY | Mean: 2.2913%, Median: 1.6204%, for a total amount of 2528 rows.
Adjusted Change for SELL | Mean: -1.1909%, Median: -0.8344%, for a total amount of 2528 rows.
Win (+) or Loss (-) | Mean: 3.4822%, Median: 2.4548%
```

<div align="justify">
<p>Building upon our awarded signals, I now have to evaluate them to get a tangible understanding. I could just stick to the training-, validation-loss, the RMSE or STD from the fine-tuning phase of the model, as shown above, but those numbers are not the best, to really grasp, how accurate my model is with it's prognoses. I achieve more understandable results by taking all predictions, with a BUY signal, and calculating the mean and median value. The same goes for the SELL signal. Combining them, results in an overall profit if positive or loss if negative. In this case, we can see that for a dataset of around 25,000 rows, we do have an (average) profit of 2.29% in our BUY signals and an (average) prevented loss of -1.19%, in our SELL signals. An overall win of 2.4% to 3.4%, depending on the calcluation methods (median or average), in a two-week time frame is a really good outcome.</p>
</div>

<br>

```
Best Predictions:
```
| Ticker | Title                                                   | Date       | Predicted Change | Future Date | Actual Change | Signal | Difference   |
|--------|---------------------------------------------------------|------------|------------------|-------------|---------------|--------|--------------|
AAPL.US | Google to allow app developers [...] | 2022-07-19 | 0.200804 | 2022-08-02 | 0.200834 | HOLD | 0.000031 |
AAPL.US | 10 Best Index Funds to Invest [...] | 2022-02-17 | -0.542589 | 2022-03-03 | -0.543218 | HOLD | 0.000628 |
INTC.US | DarkSide Hackers Mint Money [...] | 2021-05-12 | 0.571051 | 2021-05-26 | 0.571996 | HOLD | 0.000945 |
... | ... | ... | ... | ... | ... | ... | ... |

```
Worst Predictions:
```
| Ticker | Title                                                   | Date       | Predicted Change | Future Date | Actual Change | Signal | Difference   |
|--------|---------------------------------------------------------|------------|------------------|-------------|---------------|--------|--------------|
ENPH.US |	Inside IBD 50: Breakout Stocks [...] |	2022-07-15 |	-0.615360 |	2022-07-29 |	36.856678 |	HOLD |	37.472038 |
FSLR.US |	7 Stocks Set to Surge if Congress [...]	| 2022-07-19 |	-0.255714 |	2022-08-02 |	35.380423 |	HOLD |	35.636137 |
AMD.US  |	Intel's Interest in GlobalFoundries [...] |	2021-07-21 |	-0.299692 |	2021-08-04 |	31.349476 |	HOLD |	31.649168 |
... | ... | ... | ... | ... | ... | ... | ... |

```
Average Error per Ticker:
```
<table>
  <thead>
    <tr>
      <th width="550px"> Ticker </th>
      <th width="550px"> Count of Rows </th>
      <th width="550px"> Sum Errors </th>
      <th width="550px"> Average Errors </th>
    </tr>
  </thead>
  <tbody>
    <tr width="550px">
      <td align="center">PEP.US</td>
      <td align="center">391</td>
      <td align="center">488.537393</td>
      <td align="center">1.249456</td>
    </tr>
    <tr width="550px">
      <td align="center">KO.US</td>
      <td align="center">519</td>
      <td align="center">685.016616</td>
      <td align="center">1.319878</td>
    </tr>
    <tr width="550px">
      <td align="center">CL.US</td>
      <td align="center">126</td>
      <td align="center">184.063300</td>
      <td align="center">1.460820</td>
    </tr>
    <tr width="550px">
      <td align="center">...</td>
      <td align="center">...</td>
      <td align="center">...</td>
      <td align="center">...</td>
    </tr>
  </tbody>
</table>

<div align="justify">
<p>To overcome the problem that Language Models are often seen as black boxes³, since we can't really perceive why certain decisions were made, I display the best / worst decisions alongside the average error per ticker. As I directly connect the movement of a stock to certain news, I can calculate, which news articles had the smallest difference between predicted and actual change, and which ones were a total flop. This way, I can at least somewhat understand the decision-making and even maybe identify patterns in the articles or the circumstances surrounding their release, leading to more-or-less accurate predictions. Concluding with my final DataFrame containing the amount of rows, with the aggregated errors, sorted by the average error. This allows me to identify if specific stocks are more responsible for a decrease in our accuracy, than others. In the <a href="https://github.com/LennardFe/Stock-News-Analysis-with-BERT?tab=readme-ov-file#future-work">future</a>, these informations could help me adjust the algorithm by potentially removing tickers with consistently high average errors over the last period, thereby reducing errors and improving the accuracy of my model.</p>
</div>

<br>

## Challenges & Limitation
<div align="justify">
<p>Throughout the project, I was facing all kinds of problems and challenges. Some were predictable, like the fact of not having enough computing power. Some others I didn't expect. I would not have imagined struggling so much with the date format in my data. Dealing with different types of date formats proved to be more complex than anticipated, requiring research and testing in Python to effectively use and get date-related operations to work. Another point is to keep the code performant, since I've never worked on a project this size and with this big amount of data, I've never really had to worry about it. In the last three months, I've learned and adjusted my code steadily to keep it performant, readable, and as easy to understand as possible. I've tried different concepts like MultiThreading, MultiProcessing, and I've read into the topic of how to make the most efficient calculations on DataFrames. Additionally, with this size and amount of code, I was increasingly struggling to keep an overview of everything. This resulted in some deadlocks, where I was kind of overwhelmed by my own project, but also was struggling to find an entry point again, and ideas on how to progress. The two-week scheduled meetings with my supervisors were ideal to fix that, to give me new ideas and to reset my thoughts.</p>
</div>

<div align="justify">
<p>Despite my efforts to overcome these challenges, I still have to acknowledge the limitations of my project. The most obvious issue is that there are many more factors influencing stock prices than just news. Even though news represent a wide range of diverse problems and headlines, I can't expect a perfect correlation between their movement and the given article. Another concern is that my sample data, while adequate in size, only covers a time frame of two years. The evolution of news and its impact over time, and whether I can use the same model trained today in two/five/ten years, is not addressed. The potential changes in media landscape and how they might affect my model are not considered. Additionally, the current and ever-shifting mood and trust of people in the media could also change over time, impacting how individuals are influenced by news. Even if I could overcome all these issues, I could still be thrown of course by Black Swan⁴ events. The terror attacks of 9/11 or the Fukushima disaster ripped huge holes into the financial markets, unforeseen and disrupting the whole world. There are such aspects, which are unpredictable and could destroy whole investements tactics.</p>
</div>

<br>

## Conclusion

### Summary of Findings 

<div align="justify">
<p>In conclusion, my findings were generally positive, considering there were no guarantees that my theorem would even produce successful outcomes. Initially, I also included one-month and three-month targets for calculation, but they were scrapped due to the excessively large timeframe. Finding a correlation in such a large gap between news and change was quite unlikely, so i focused on three shorter targets instead: <b>Three days</b> resulting in an average profit of <b>0.64%</b>, <b>one week</b> with a profit of <b>1.06%</b>, and <b>two weeks</b> with a surprisingly large profit of <b>3.48%</b>; These numbers should be approached cautiously as achieving them or even coming close to them in a real-life scenario is quite unlikely. When comparing my model to the return of the MSCI World ETF, a passive investment strategy which averages around 10% per year⁵, investing in my model with a neutral (average) 1% profit every two weeks could potentially surpass the MSCI World return. For example, consider an initial investment of 100€:</p>
</div>

<p><li><b>MSCI World ETF Strategy</b> - AVG Return of 10% per Year - 1 Year: 110€, 2 Years: 121€, 3 Years: 133.10€</li>
<li><b>My Model Strategy</b> - AVG Return of 1% per two Weeks - 1 Year: 126€, 2 Years: 152.52€, 3 Years: 192.29€</li></p>


<div align="justify">
<p>Completing this prototype, with still a lot of room for potential improvement, and achieving these results, represents a significant milestone. To further validate and solidify my theory, it would be necessary to test and work on larger datasets and explore other settings. </p>
</div>


### Future Work

<div align="justify">
<p>Despite being happy with the current and, for the time being, final state of the prototype, I wasn't able to implement every idea I had throughout this project. Some ideas that arose couldn't be realized due to the close deadline or would have required significant and time-consuming changes. Therefore, I would like to at least list the ideas that were not implemented:</p>
</div>

<div align="justify">
<p><li>I already addressed the topic of training my models on so few epochs earlier. Using only three epochs isn't the best approach. It's not the worst, since BERT as it is, remains a strong model, but more epochs with more intensive research on which parameters to use would definitely elevate this model to a whole new level. Due to my limitations in time and resources, this wasn't possible, but it would definitely be one of the first steps in the future.</li></p>
</div>

<div align="justify">
<p><li>As mentioned earlier, I'm unsure about whether a model trained and tested on data from the years 2021 and 2022 can be effectively applied to different time periods. It's unclear whether the accuracy would remain close or if the relationship between news and price movements has changed over time. Taking a deeper dive into this topic, could reveal valuable insights into the dynamics of news sentiment and its impact on the market, across various time periods.</li></p>
</div>

<div align="justify">
<p><li>Additionally, many news articles discuss more than one stock, comparing and evaluating their current states individually, this could result in varying sentiments in one article regarding different stocks. One way to address this would be to identify relevant passages per ticker, for instance, through entity recognition tools like those provided in the SpaCy and NLTK libraries. Combining this with the <i>cleannames</i> from the Leeway fundamentals could enable me to retrieve relevant parts per ticker, analyze them and predict a change individually, instead of treating the entire article as a single unit.</li></p>
</div>

<div align="justify">
<p><li>Instead of deciding between signals per news alone, we could aggregate all predictions made on one day for each stock and combine them into an overall prediction, with an assigned, potentially more stable, signal. Also helping to reduce the risk of outliers.</li></p>
</div>

<div align="justify">
<p><li>Currently, I'm splitting the data into the train, test, and validation sets by grouping them by ticker and then dividing the news equally among the three sets. This approach ensures a good representation of each ticker in the datasets but could potentially lead to overfitting. The model might learn to rely solely on the patterns specific to, for example, AAPL news and generalize poorly to news from other tickers. This could result in inaccuracies when predicting the movements of stocks not present in the training data, as the model wouldn't have learned broader patterns in news structure beyond these individual tickers.</li></p>
</div>

<div align="justify">
<p><li>Some news articles summarize their content at the beginning. What about just using the first few sentences combined with the title of the corresponding article? This would allow me to drastically reduce the size of the model and the duration of training, as well as the time it takes to make predictions, since we wouldn't need to process such long articles anymore. A model could be trained in this manner and then compared to a model that makes predictions based on the entire article. Depending on the accuracy of the smaller model, a minor loss in it could be worth trading for better performance and faster execution.</li></p>
</div>

<div align="justify">
<p><li>To compare all the different combinations of parameters set in the <a href="https://github.com/LennardFe/Stock-News-Analysis-with-BERT/blob/main/executor.py">executor.py</a> file, I could loop over various values and execute the notebook multiple times. In this scenario, I could easily compare all the different settings and their results with only one execution To achieve that, I would need to adjust the code accordingly or create a whole new file with a pool of variables to select from. For this case it would be more rational, to just display the results and not the whole code, this would make it easier to follow and compare.</li></p>
</div>

<div align="justify">
<p><li>Lastly, I'm not quite satisfied with the way I save and load the retrieved data to the MongoDB.  Even though this isn't a new idea or approach in terms of the analytical and functional aspects of the project, I still wanted to address it. I currently combine the ticker and time period into a single name and then split it into smaller batches, as MongoDB is more efficient with smaller sets.   An example would look like this:  <i>AAPL.US_01.01.2020_01.01.2021_X</i>, where the "X" serves as a placeholder for the different subsets, the original data is split into. For instance, if we have 10,000 news articles and a batch size of 2,000, we would split them into 5 subsets.  The number of subsets is stored in the given construced name, just without any batch identifier at the end, e.g.  <i>AAPL.US_01.01.2020_01.01.2021</i>, so the program knows how many batches it has to retrieve. The issue I'm dissatisfied with is that if we want to retrieve news for a different time period than used above, for example, in between January and February of 2020, even though the data is available inside the time frame of January 2020 to January 2021, the program won't find it and will fetch and save the data again. The same applies to trading days and stock values, since the method of saving and loading data from the database is nearly the same overall.</li></p>
</div>

<br>

## References

1. Ahlgren, Oskar. "Research on Sentiment Analysis: The First Decade". [2016] 
	- https://sentic.net/sentire2016ahlgren.pdf 
2. Vaswani, Ashish et. al. "Attention Is All You Need". [2017]
	- https://arxiv.org/pdf/1706.03762.pdf
3. Lasserre, Pat. "Making Large Language Models More Explainable". [2023]
	- https://gsitechnology.com/making-large-language-models-more-explainable/
4. Investopedia. "Black Swan in the Stock Market: What Is It, With Examples and History". [2023]
	- https://www.investopedia.com/terms/b/blackswan.asp
5. Curvo, "Historical performance of the MSCI World index". [2024]
	- https://curvo.eu/backtest/en/market-index/msci-world?currency=eur
