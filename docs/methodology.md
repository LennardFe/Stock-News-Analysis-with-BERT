# Methodology

<div align="justify">
<p>  
This methodology is structured to guide through the process of going from raw data to the final result of predicting stock movements based on news articles. I've broken down the project into six key steps and one additional step. Each step covers the functions associated with it, how to use them, their functionality, and the expected results. The focus is on the main functionality of the project. Because of that, not every function and its corresponding docstrings are explicitly mentioned or fully displayed here. If code has been cut from the methodology, you'll see this "<i>[...]</i>". For a full explanation of each function, refer to the docstrings in the main <a href="https://github.com/LennardFe/Stock-News-Analysis-with-BERT/blob/main/notebook.ipynb">notebook</a>. </p>
</div>

<br>

## Table of Contents
1. [Prepatory Work](#1-prepatory-work)
2. [Data Collection](#2-data-collection)
3. [Data Preperation](#3-data-preperation)
4. [Finetuning LLM](#4-finetuning-llm)
5. [Sampling (Optional)](#5-sampling-optional)
6. [Prediction and Signals](#6-prediction-and-signals)
7. [Performance Evaluation](#7-performance-evaluation)

<br>

## 1. Prepatory Work
<div align="justify">
Before I started working with the Leeway API, I used the <a href="https://www.alphavantage.co/">AlphaVantage</a> API to get a feeling for python in a data analytics context. This involved converting the API Response into usable python format and plotting graphs using pyplot and seaborn. All in all quite simple steps, to get more familiar with python as a language and my ability to get human-friendly informations out of hard to understand numeric data.
</div>

<br>

## 2. Data Collection

### Extract stocks from ETFs
<div align="justify">
<p>I begin by extracting the stocks from the given ETFs and saving them into a separate DataFrame. The concept of a DataFrame is essential for the entire project, it's a structure that organizes data into a two-dimensional table, which allows efficient and flexible operations on it. We call the Leeway-API through the self-declared <i>leeway</i> function and convert the json-response into a more usable DataFrame.</p>
</div>

**Function:** 
```python
def extract_from_etf(etfs):
    """
    Extracts stocks from a list of ETFs and combine them into a DataFrame.

    Args:
    - etfs (list): List of ETF symbols.

    Returns:
    - DataFrame: A DataFrame containing stocks for the specified ETFs, including columns "ticker" and "etf".
    """
    df = pd.DataFrame() 
    for etf in etfs:   
        df_fundamental = pd.DataFrame()
        json_fundamental = leeway("FUNDAMENTALS", etf)

        df_fundamental["ticker"] = json_to_df((json_fundamental["ETF_Data"])["Holdings"])
        df_fundamental["etf"] = etf

        df = pd.concat([df, df_fundamental], ignore_index=True)
    return df
```

**Usage:** 
```python
#Params:
etfs 		= ["XLK.US", "XLP.US"]

# Function Call:
stocks_df 	= extract_from_etf(etfs)
```

**Returns:** 
```
Type:		Pandas.DataFrame
Structure:	[ ticker | etf ]
Content:	All stocks the ETFs XLK and XLP are consisting of.
```


### Retrieve valid trading days

<div align="justify">
<p>The next step is to retrieve all open trading days for the given Stock Exchange over the chosen time frame. For this purpose, we utilize the Pandas-Marketcalendar package. For my prototype, I worked with the NYSE. In this and following functions, I save the request to a MongoDB. This helps reducing the amount of API-requests to speed up the code.</p>
</div>

**Function:** 
```python
def get_trading_days(start_date, end_date, calendar):
    """
    Get trading days from the specified calendar between the specified start and end dates.

    Args:
    - start_date (str): Start date in the format "YYYY-MM-DD".
    - end_date (str): End date in the format "YYYY-MM-DD".
    - calendar (str): The name of the calendar to retrieve trading days from.

    Returns:
    - str: JSON-formatted string containing a list of trading days.
    """
    adj_end_date = (pd.to_datetime(end_date) + pd.DateOffset(months=3)).strftime("%Y-%m-%d") # add 3 months to the end date to make sure we get all trading days , TODO: magic number

    name = f"{calendar}_{start_date}_{adj_end_date}_market_open"

    # check if the data is already in the db, if not fetch it from the api
    if not load_from_db(name):
        stock_exchange= mcal.get_calendar(calendar) 
        schedule = stock_exchange.schedule(start_date, adj_end_date)
        trading_days = schedule["market_open"].dt.date.unique()

        dates_str = [d.isoformat() for d in trading_days]
        json_dates = json.dumps(dates_str)

        save_to_db(json_dates, name)
    return load_from_db(name)
```

**Usage:** 
```python
#Params:
start_date 	= "2021-01-01"
end_date 	= "2022-12-31"
calendar 	= StockExchange.NYSE 

# Function Call:
dates_js 	= get_trading_days(start_date, end_date, calendar)
```

**Returns:** 
```
Type:		JSON-structured array string
Structure:	'["date1", "date2", "date3", ...]'
Content:	All valid trading days between first of January 2021 and the last day of 2022
```

### Obtain Stock, ETF and Marktecap values

<div align="justify">
<p>I use the extracted stocks from above, in the format of the DataFrame, to retrieve all stock, ETF, and market cap values per row (stock) of the DataFrame over the whole given time period. I make sure to make as few requests to the API as possible and also multithread the requests to reduce execution time.</p>
</div>

**Function:** 
```python
def get_values(df, start_date, end_date):
    """
    Retrieves stock, ETF, and market capitalization values for given tickers and ETFs within a specified date range.

    Args:
    - df (DataFrame): Pandas DataFrame containing data with necessary columns "etf" and "ticker".
    - start_date (str): Start date of the data retrieval period (formatted as "YYYY-MM-DD").
    - end_date (str): End date of the data retrieval period (formatted as "YYYY-MM-DD").

    Returns:
    - dict: A dictionary containing combined stock, ETF, and market capitalization values with dates as keys.
    """
    session = create_session()

    # add 3 months to the end date, so we get all values including these from the future date. #TODO: magic number
    adj_end_date = (pd.to_datetime(end_date) + pd.DateOffset(months=3)).strftime("%Y-%m-%d")

    # get unique values from the dataframe
    unique_etf_values = df["etf"].unique()
    unique_ticker_values = df["ticker"].unique()

    # get the stock,etf and mc values from the api using multithreading
    with concurrent.futures.ThreadPoolExecutor() as executor:
        stock_data = list(executor.map(lambda ticker: get_value_per_row(ticker, start_date, adj_end_date, session), unique_ticker_values))
        etf_data = list(executor.map(lambda etf: get_value_per_row(etf, start_date, adj_end_date, session), unique_etf_values))
        mc_data = list(executor.map(lambda ticker: get_marketcap_per_row(ticker, start_date, adj_end_date, session), unique_ticker_values))

    stock_prices = collect_valid_data(stock_data)
    stock_prices.update(collect_valid_data(etf_data))
    stock_prices.update(collect_valid_data(mc_data))

    return stock_prices

def get_value_per_row(ticker, start_date, end_date, session):
    """
    Fetches stock or ETF values for a given ticker from either the API or database.
	[...]
    """
    name = f"{ticker}_{start_date}_{end_date}_values" 

    # check if the data is already in the db, if not fetch it from the api
    if not load_from_db(name):
        try:
            value_json = leeway("VALUE", ticker, start_date, end_date, session=session)
            dict = { ticker: { entry["date"]: entry["adjusted_close"] for entry in value_json } }
        except Exception as e:
            print(f"Error fetching stock values: {e}")
            dict = {}

        save_to_db(dict, name)
    return load_from_db(name)

def get_marketcap_per_row(ticker, start_date, end_date, session): [...]

def collect_valid_data(data): [...]

```

**Usage:** 
```python
#Params:
start_date 	= "2021-01-01"
end_date 	= "2022-12-31"
stocks_df 	= extract_from_etf(["XLK.US"])

# Function Call:
values_js 	= get_values(stocks_df, start_date, end_date)
```

**Returns:** 
```
Type:		Python dictionary with key-value pairs
Structure:	{'ticker1': {'date': value', ...}, 'ticker2' {'date': value', ...}, ...}
Content:	All corresponding stock, XLK-ETF and marketcap values for each trading day in 2021/2022.
```


### Get News articles 

<div align="justify">
<p>  I use the same parameters as those employed in the function before. I retrieve news articles for each stock within our specified time frame. To speed up the process of API-calls I resort back to multithreading. Additionally, I store the loaded news in MongoDB by dividing them into batches, as MongoDB is more efficient with smaller sets. A workaround was necessary to work with the API's limitation of a maximum of 1000 responses per request, as the number of news articles per stock within the given time frame exceeds the 1000 limit.  I achieve this by iterating over the first fetched data and checking if the news articles span the entire specified time frame. If not, I send a new request to the API to retrieve the remaining data. I repeat this process until the entire time frame is covered.</p>
</div>

**Function:** 
```python
def get_content(df, start_date, end_date):
    """
    Retrieves and combines news for a DataFrame containing tickers.

    Args:
    - df (DataFrame): Pandas DataFrame containing relevant data with columns "ticker" and "etf".
    - start_date (str): Start date of the data retrieval period (formatted as "YYYY-MM-DD").
    - end_date (str): End date of the data retrieval period (formatted as "YYYY-MM-DD").

    Returns:
    - DataFrame: A DataFrame containing combined news, including columns for "ticker" and "etf".
    """
    # apply the retrieve_and_combine_news function, which provied a walkaround the 1000 limit of the api
    with concurrent.futures.ThreadPoolExecutor() as executor:
        news = list(executor.map(lambda row: retrieve_and_combine_news(row, start_date, end_date), df.to_dict(orient="records")))
    
    final_df = pd.concat(news, ignore_index=True)

    format_final_df = format_date(final_df)  # format date to YYYY-MM-DD

    return format_final_df

def retrieve_and_combine_news(row, start_date, end_date):
    """
    Helper function to retrieve and combine news with ticker and etf symbol for each row in a DataFrame.
	[...]
    """
    # call the get_news function, which provides a workaround for the 1000 limit of the API
    news_df = get_news_per_row(row["ticker"], start_date, end_date)
    # add new column to the df to show the matching stock and etf
    news_df = news_df.assign(ticker=row["ticker"], etf=row["etf"])

    return news_df

def get_news_per_row(ticker, start_date, end_date):
    """
    Retrieves news data for a given stock ticker from the API with a workaround for the 1000-answer limit.
	[...]
    """
    name = f"{ticker}_{start_date}_{end_date}_news"

    if not load_from_db(name):
        # get data from api and load them into dataframe
        news_json = leeway("NEWS", ticker, start_date, end_date) 
        news_df = json_to_df(news_json, ["title", "date", "content"])

        # change sorting so the oldest date is on top
        if not news_df.empty: 
            news_df = news_df.sort_values(by="date", ascending=True)

            # load the oldest date from the dataframe into the first_date variable
            date_obj = datetime.fromisoformat(str(news_df["date"].iloc[0]))
            first_date = date_obj.strftime("%Y-%m-%d") 

            # check if first_date equals our start_date parameter, if not that means the news data exceeds the 1000 limit
            while first_date != start_date:
                # new request to api with our old start_date data and the time_to data, which is now our oldest date in the dataframe
                news_json = leeway("NEWS", ticker, start_date, first_date)
                new_news_df = json_to_df(news_json, ["title", "date", "content"])

                # also sort this new df so the oldest date is on top, only if new df exists
                if not new_news_df.empty:
                    new_news_df = new_news_df.sort_values(by="date", ascending=True)

                # combine the two dfs
                news_df = pd.concat([new_news_df, news_df])     

                # check if the "old" oldest date equals the "new" oldest date, if so we can leave the loop since we dont have any new data
                date_obj = datetime.fromisoformat(str(news_df["date"].iloc[0]))

                if(first_date == date_obj.strftime("%Y-%m-%d")):
                    break
                
                first_date = date_obj.strftime("%Y-%m-%d")

                # we can either leave the loop by having the same oldest date for two iterations or if the oldest date equals our time from parameter

        # drop duplicates
        news_df.drop_duplicates(subset=["title"], inplace=True)

        return split_data_to_db(news_df, name, 2000)

    else:
        return combine_data_from_db(name)
```

**Usage:** 
```python
#Params:
start_date 	= "2021-01-01"
end_date 	= "2022-12-31"
stocks_df 	= extract_from_etf(["XLK.US"])

# Function Call:
news_df 	= get_content(stocks_df, start_date, end_date)
```

**Returns:** 
```
Type:		Pandas.DataFrame
Structure:	[ title | date | content | ticker | etf ] 
Content:	All news articles in the years 2021/2022 for every stock from the ETF XLK.
```

<br>

## 3. Data Preperation

### Set dates to nearest trading day

<div align="justify">
<p>I use the fetched news, valid trading days, and the newly set target to update the release dates of the news article to the nearest valid trading day before the release of the article. Next to this step, I also add the future target date based on the specified target-parameter. If the date happens to be on a day when the stock exchange is closed, we also move to a valid trading day before that. For the particular case where there are no valid trading days before our target date, we adjust to the closest date overall.</p>
</div>

**Function:** 
```python
def update_dates(df, dates_js, target):
    """
    Update date columns in a DataFrame with the nearest dates from a given list.

    Args:
    - df (DataFrame): Input DataFrame.
    - dates_js (str): JSON-formatted string containing a list of valid dates.
    - target (str): Specified target.

    Returns:
    - DataFrame: DataFrame with updated date columns.
    """
    df_copy = df.copy() # copy so we dont change the original df
    df_copy["date"] = pd.to_datetime(df_copy["date"])
    relevant_columns = ["date", f"date_{target}"] # only keep the relevant columns, date and the target dates

    target_mapping = {"3D": 3, "1W": 1, "2W": 2, "1M": 1, "3M": 3}
    target_value = target_mapping.get(target)

    if "M" in target:     df_copy[f"date_{target}"] = df_copy["date"] + pd.DateOffset(months=target_value)
    elif "W" in target:   df_copy[f"date_{target}"] = df_copy["date"] + pd.DateOffset(weeks=target_value)
    else:                 df_copy[f"date_{target}"] = df_copy["date"] + pd.DateOffset(days=target_value)

    dates_series = pd.to_datetime(pd.Series(eval(dates_js)))  # convert JSON string to pandas datetime series

    for col in relevant_columns: # only call the function if the date is not in the list of valid dates
        df_copy[col] = df_copy[col].apply(lambda x: find_nearest_date_before(x, dates_series) if x not in dates_series.values else x)

    # format the dates to YYYY-MM-DD
    format_df = format_date(df_copy, relevant_columns)  

    return format_df

def find_nearest_date_before(date, date_list):
    """
    Find the nearest valid date before a given date in a list.
	[...]
    """
    date = pd.to_datetime(date)
    valid_dates_before = [x for x in date_list if x < date] # take only dates before given date

    if not valid_dates_before:
        # if there are no valid dates before the date, return the closest date
        nearest_date = min(date_list, key=lambda x: abs(date - x))
        return nearest_date
    
    nearest_date = max(valid_dates_before) # choose closest date before given one
    return nearest_date
```

**Usage:** 
```python
#Params:
news_df		= get_content(stocks_df, start_date, end_date) # as before in point 2
dates_js	= get_trading_days(start_date, end_date, calendar) # as before in point 2
target 		= Target.W1

# Function Call:
up_news_df 	= update_dates(news_df, dates_js, target)
```

**Returns:** 
```
Type:		Pandas.DataFrame 
Structure:	[ title | date | content | ticker | etf | date_1W ]
Content:	Updated the release dates and add date for one week in the future, based on trading days.
```


### Calculate changes based on the target

<div align="justify">
<p>Since I've updated the release and target dates, according to the calendar, we can now fetch the values of the stocks on these particular days without worrying that the stock exchange might not have been open. I do that using our retrieved values from earlier and calculate the change in value of the stock from the release to the target date. I adjust the change based on the overall change of the corresponding ETF. Additionally, I include the marketcap value for each row.</p>
</div>

**Function:** 
```python
def calc_changes(df, values_js, target):
    """
    Calculate adjusted changes for the given df and future target.

    Args:
    - df (DataFrame): Input DataFrame with news and dates.
    - values_js (str): JSON-formatted string containing stock, ETF, and marketcap values.
    - target (str): target for future date calculations.

    Returns:
    - DataFrame: DataFrame with alculated changes.
    """
    df_copy = df.copy() # copy so we dont change the original df
    df_copy[[f"change_{target}", "marketcap"]] = df_copy.apply(lambda row: calculate_changes_per_row(row, values_js, target), axis=1)

    return df_copy

def calculate_changes_per_row(row, values_js, target):
    """
    Call function for calculation of adjusted changes and marketcap for each row in a DataFrame.
	[...]
    """
    etf = row["etf"]
    date = row["date"]
    ticker = row["ticker"]
    target_date = row[f"date_{target}"]
    ticker_mc = f"{ticker}_mc"

    # calculate the adjusted change for each row and add the marketcap column
    change = calculate_adjusted_change(ticker, etf, date, target_date, values_js)
    marketcap = values_js[ticker_mc].get(date)

    return pd.Series({f"change_{target}": change, "marketcap": marketcap})

def calculate_adjusted_change(ticker, etf, date, target_date, values_js):
    """
    Calculate adjusted change for a specific stock and ETF.
	[...]
    """
    adjusted_change = None

    # get the values from the dictionary
    stock_values = values_js[ticker]
    etf_values = values_js[etf]

    # check if the dates are in the dictionary
    if (date in stock_values) and (target_date in stock_values) and (date in etf_values) and (target_date in etf_values):

        stock_original_value = stock_values[date]
        stock_future_value = stock_values[target_date]

        etf_original_value = etf_values[date]
        etf_future_value = etf_values[target_date]

        stock_change = ((stock_future_value - stock_original_value) / stock_original_value) * 100
        etf_change = ((etf_future_value - etf_original_value) / etf_original_value) * 100

        adjusted_change = stock_change - etf_change

    return adjusted_change
```

**Usage:** 
```python
#Params:
up_news_df	= update_dates(news_df, dates_js, target) # as before
values_js	= get_values(stocks_df, start_date, end_date) # as before in point 2
target		= Target.W1

# Function Call:
changes_df 	= calc_changes(up_news_df, values_js, target)
```

**Returns:** 
```
Type:		Pandas.DataFrame 
Structure:	[ title | date | content | ticker | etf | date_1W | change_1W | marketcap ]
Content:	Calculate the change of the stock from each news, from release to the target of one week.
```


### Preprocess the data

<div align="justify">
<p> I apply various steps to process the data of the content column from the given DataFrame, including the removal of stop words and symbols, splitting rows if the size of the content exceeds the BERT maximum character length, dropping NaN values and duplicates. We don't have to manually set everything to lowercase, as most language models do that with their tokenizer. Additionally, I added a stemmer and lemmatizer, even though they are currently not used in the trained models, they do function and can be used in future work.</p>
</div>

**Function:** 
```python
def preprocess(df):
    """
    Perform pre-processing steps on a DataFrame by calling different functions.

    Args:
    - df (DataFrame): Input DataFrame.

    Returns:
    - DataFrame: Processed DataFrame after applying various pre-processing steps.
    """
    df_copy = df.copy()
    
    # drop nan values
    df_copy.dropna(inplace=True)

    # remove symbols, punctuations
    df_copy["content"] = df_copy["content"].apply(remove_symbols)

    # remove stop words
    df_copy["content"] = df_copy["content"].apply(remove_stops)

    # split text by the maximum limit of bert
    df_copy = split_text(df_copy, column="content", limit=512)

    # since different news articles with differenct titles can have the same content, we drop duplicates
    df_copy = df_copy.drop_duplicates(subset=["content", "ticker"]) 

    # sort by ticker, then date
    df_copy.sort_values(by=["ticker", "date"], inplace=True)

    return df_copy

def split_text(df, column, limit): [...]

def count_words(text): [...]

def remove_stops(text): [...]

def remove_symbols(text): [...]

def lemmatize_text(text): [...]

def stem_text(text): [...]
```

**Usage:** 
```python
#Params:
changes_df 	= calc_changes(up_news_df, values_js, target) # as before

# Function Call:
prep_df 	= preprocess(changes_df)
```

**Returns:** 
```
Type:		Pandas.DataFrame 
Structure:	[ title | date | content | ticker | etf | date_1W | change_1W | marketcap ]
Content:	The data of the content column gets updated with the preprocessing steps.
```


### Split data by ticker

<div align="justify">
<p>I implemented this specific split of the training, validation, and testing data to ensure that each one of these datasets has the same or at least similar representation. If I were to just split the data at random, I could end up with a case where a majority of news from the "Magnificent Seven" (Alphabet, Amazon, Apple, Meta, Microsoft, Nvidia, Tesla) end up in the training dataset. Since these stocks are special in their general perception in the media and their price movements, I have to ensure that we divide them equally in each dataset so my model won't have a bias based on the BIG7 or maybe even other stocks. I group the DataFrame by ticker and split it afterwards. In this case, each stock has 70% of its news in the training data, 15% in the validation data, and the remaining 15% in the test data. This way, I maintain the same percentage representation for all stocks in each dataset.</p>
</div>

**Function:** 
```python
def split_df_by_ticker(df):
    """
    Splits a DataFrame into training, testing, and validation sets based on ticker symbols.

    Args:
    - df (DataFrame): Pandas DataFrame containing data with a "ticker" column.

    Returns:
    - tuple of DataFrames: Training, testing, and validation sets.
    """
    train_set = pd.DataFrame()
    test_set = pd.DataFrame()
    val_set = pd.DataFrame()

    for _, group_data in df.groupby("ticker"):
        train, test_and_val = train_test_split(group_data, test_size=0.3, random_state=42) # split each group by 70/30
        test, val = train_test_split(test_and_val, test_size=0.5, random_state=42) # split the test_val in 50/50

        # concat the splits to the final sets
        train_set = pd.concat([train_set, train])
        test_set = pd.concat([test_set, test])
        val_set = pd.concat([val_set, val])

    print(f"Trainingsset: {len(train_set)} Zeilen")
    print(f"Testset: {len(test_set)} Zeilen")
    print(f"Validierungsset: {len(val_set)} Zeilen")

    return train_set, test_set, val_set
```

**Usage:** 
```python
#Params:
prep_df		 = preprocess(changes_df) # as before

# Function Call:
train, test, val = split_df_by_ticker(prep_df)
```

**Returns:** 
```
Type:		3x Pandas.DataFrame 
Structure:	[ title | date | content | ticker | etf | date_1W | change_1W | marketcap ]
Content:	The DataFrame gets split into three subsets with the same structure, but different data.
```

**Prints:** 
```
The amount of rows per newly split dataset. 
```


### Obtain labels for the specified interval

<div align="justify">
<p>To maintain a dynamic usability, where I don't have to adjust the column names every time I change parameters, I've offloaded this process to a separate function which maps the input target to their corresponding column names. While not directly necessary, this approach helps with the consistency and reduces the potential for errors in the code.</p>
</div>

**Function:** 
```python
def get_label_for_target(target):
    """
    Get label information for the specified target date.

    Args:
    - target (str): Target date identifier (e.g., "3D", "1W", 2W", "1M", "3M").

    Returns:
    - tuple: A tuple containing the content column name, change column name and date column name.

    Raises:
    - ValueError: If the provided target is not valid.
    """
    
    target_mappings = {
        "3D": ("content", "change_3D", "date_3D"),
        "1W": ("content", "change_1W", "date_1W"),
        "2W": ("content", "change_2W", "date_2W"),
        "1M": ("content", "change_1M", "date_1M"),
        "3M": ("content", "change_3M", "date_3M")
    }

    if target not in target_mappings:
        raise ValueError("Invalid target")

    return target_mappings[target]
```

**Usage:** 
```python
# Params:
target = Target.W1

# Function Call:
text, label, label_date = get_label_for_target(target)
```

**Returns:** 
```
Type:		Tuple
Structure:	Content, Change, Date column names
Content:	The corresponding content, change and date column names for the one week target.
```


### Adjust column names and drop columns

<div align="justify">
<p>To conclude the preparation of the data, I adjust the names of the columns and potentially drop some columns that are no longer needed. I rename the <i>content</i> column to "text" and the <i>change_{TARGET}</i> to "label" to fit the input conventions of the language model. For testing and evaluation purposes, we might include the <i>date_{TARGET}</i>, renamed to "label_date", and other columns to better understand the decision-making of the model.</p>
</div>

**Function:** 
```python
def adjust(df, text, label, date=None):
    """
    Adjusts the columns of a DataFrame to standard names with optional date column with a new label.

    Args:
    - df (DataFrame): Pandas DataFrame to be adjusted.
    - text (str): Name of the column representing textual data.
    - label (str): Name of the column representing labels.
    - date (str, optional): Name of the column representing dates.

    Returns:
    - DataFrame: Adjusted DataFrame with columns named "text" and "label", and an optional "label_date" column.
    """
    df_copy = df.copy()
    
    # rename the columns to text and label
    if date is None:
        df_copy = df[[text, label]]
        return df_copy.rename(columns= {text: "text", label: "label"})

    df_renamed = df_copy.rename(columns= {text: "text", label: "label", date: "label_date"})

    return df_renamed
```

**Usage:** 
```python
# Params:
train, test, val = split_df_by_ticker(prep_df) # as before
text, label, label_date = get_label_for_target(target) # as before

# Function Call:
train		= adjust(train, text, label)
test		= adjust(test, text, label, label_date)
```

**Returns:** 
```
Type:		Pandas.DataFrame 

Train (Date is None):
Structure:	[ text | label ]
Content:	The content, change_1W columns renamed to text, label. Other columns get dropped.

Test (Date is not None):
Structure:	[ title | date | text | ticker | etf | label_date | label | marketcap ]
Content:	The content, date_1W, change_1W columns renamed to text, label_date and label. 
```
<br>


## 4. Finetuning LLM

<div align="justify">
<p>As a base for my project, I worked and also tested on different BERT models. There are many more language models (LLMs) that would be fitting for this task and may even be better suited for it, like the quite new LLAMA 2. The biggest reason why I stuck to BERT is the amount of tutorials, assistance, and external examples available. Since BERT is one of the older LLMs, there are many resources to choose from, which is ideal for a beginner like me. Additionally, a lot of community-made BERT models are available, which reduce the overall size of it, while still trying to keep the performance close to the original model. This was also important to me, since I don't have unlimited or even powerful enough hardware resources for these long and demanding calculations. </p>
</div>

<div align="justify">
<p>I begin by tokenizing the training and validation datasets and loading the specified BERT model, with an additional sequence classification layer, which is recommended and partly necessary to make classification tasks. I set the metrics that are relevant for understanding the precision of the model and are also crucial for choosing the best epoch with its model in the end. Through the parameters, I am able to specify the most important settings, which are then fully initialized in the Training Arguments, along with other options. The last step concludes with setting the optimizer, training the model, and saving it to the output directory. Obviously, if the file already exists, the model will be loaded and not trained again.</p>
</div>

<div align="justify">
<p>For fine-tuning the BERT model, I stick to the HuggingFace library with its easy and fast-to-use transformers. This allows me to easily set up the structure for working with a wide range of pre-trained models in the horizon of NLP tasks and LLMs. This removes the need to manually create training loops and makes the whole process more straightforward. Other options would be to directly use PyTorch, which Huggingface is built upon, or TensorFlow.</p>
</div>

**Function:** 
```python
def bert(model_name, train_df, val_df, model_type, learning_rate, epochs, batch_size, weight_decay):
    """
    Fine-tunes a BERT-based model on the given training dataset and evaluates it on the validation dataset.

    Args:
    - model_name (str): The name of the pre-trained BERT model to use.
    - train_df (pandas.DataFrame): The training dataset as a pandas DataFrame.
    - val_df (pandas.DataFrame): The validation dataset as a pandas DataFrame.
    - model_type (str): The type of model to use (DistilBERT, FinBERT, ...).
    - learning_rate (float): Learning rate for the model.
    - epochs (int): Number of epochs to train the model.
    - batch_size (int): Batch size for the model.
    - weight_decay (float): Weight decay parameter for the optimizer.

    Returns:
    - transformers.AutoModelForSequenceClassification: The fine-tuned BERT model.
    """
    # combining model_name to directory
    output_directory = "models"
    model_directory = os.path.join(output_directory, model_name)

    # if the folder doesnt exist, create it	
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # if the model doesnt already exists, train it
    if not os.path.exists(model_directory):

        # load datasets from df
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)

        # load the tokenizer from the pretrained bert model
        tokenizer = AutoTokenizer.from_pretrained(model_type)

        # tokenizer function which "tokenizes" the input so the model can work with it
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)

        # call tokenize function through map of our dataset (high level citizien)
        train_tokenized = train_dataset.map(tokenize_function, batched=True)
        val_tokenized = val_dataset.map(tokenize_function, batched=True)

        # load the bert model for sequence classification, which has one more layer compared to the regular 
        model = AutoModelForSequenceClassification.from_pretrained(model_type, num_labels=1)

        # rmse and standard deviation
        def compute_metrics_regr(eval_pred):
            predictions, labels = eval_pred
            rmse = mean_squared_error(labels, predictions, squared=False)
            labels_std = np.std(labels)
            rmse_std = ((rmse - labels_std) / labels_std)
            return {"rmse": rmse, "std": labels_std, "(rmse-std)/std": rmse_std}

        # trainer arguments
        training_args = TrainingArguments(
            output_dir=model_directory,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            logging_strategy="epoch",
            logging_steps=1,
            load_best_model_at_end=True,
            metric_for_best_model="(rmse-std)/std")

        # create trainer object with parameters
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=val_tokenized,
            compute_metrics=compute_metrics_regr,
        )

        optimizer = torch.optim.AdamW(
            trainer.model.parameters(),
            lr=training_args.learning_rate,
            weight_decay=training_args.weight_decay
        )

        # set the optimizer in the trainer
        trainer.optimizer = optimizer

        # finetune the model
        trainer.train()

        # save our finetuned model to output path
        trainer.save_model(model_directory)

    # if the file already exists then load the model
    finetuned_model = AutoModelForSequenceClassification.from_pretrained(model_directory)
    return finetuned_model
```

**Usage:** 
```python
# Params:
train		= adjust(train, text, label) # as before in point 3
val		= adjust(val, text, label) # as before in point 3
model_name	= "DBERT_1W_V1"
model_type	= "distilbert-base-uncased"
learning_rate 	= 1e-5
epochs		= 3
batch_size	= 16
weight_decay  	= 0.015

# Function Call:
model		= bert(model_name, train, val, model_type, learning_rate, epochs, batch_size, weight_decay)
```

**Returns:** 
```
Type:		Transformers.AutoModelForSequenceClassification 
Structure:	Folder containing the checkpoints after each epoch, with overall configs and arguments.
Content:	The fine-tuned BERT model, based on given parameters and training dataset.
```
<br>


## 5. Sampling (Optional)

<div align="justify">
<p>With the aim of reducing execution time, while still maintaining an equal and representative split of the test data, we group by the ticker column and select the same amount of rows per ticker. Since some tickers have fewer news-articles than others, we take the minimum between the desired rows and available rows to prevent creating duplicates in our new test data. This step is optional, since we could specify the total rows as none or simply don't execute this function at all, we would continue with the whole test dataset.</p>
</div>

**Function:** 
```python
def sample_rows(df, total_rows, random_seed=None):
    """
    Sample rows from a DataFrame for each ticker to maintain an equal representation.

    Args:
    - df (DataFrame): The input DataFrame.
    - total_rows (int): The total number of rows in the sampled DataFrame.
    - random_seed (int, optional): Seed for reproducibility.

    Returns:
    - sampled_df (DataFrame): The sampled DataFrame.
    """
    if total_rows is None: # if specified, use all rows
        return df

    np.random.seed(random_seed)  # seed for reproducibility
    desired_rows_per_ticker = total_rows / len(df["ticker"].unique())
    
    def sample_group(group):
        ticker_rows = group.shape[0]
        
        # calculate the number of rows to sample
        sampled_rows = int(min(desired_rows_per_ticker, ticker_rows))
        
        # sample w/o replacement
        sampled_group = group.sample(sampled_rows, replace=False)
        
        return sampled_group

    sampled_df = df.groupby("ticker").apply(sample_group).reset_index(drop=True)

    return sampled_df
```

**Usage:** 
```python
# Params:
test		= adjust(test, text, label, label_date) # as before in point 3
sample_size	= 1000
random_seed	= 43

# Function Call:
small_test	=  sample_rows(test, sample_size, random_seed)
```

**Returns:** 
```
Type:		Pandas.DataFrame 
Structure:	[ title | date | content | ticker | etf | date_1W | change_1W | marketcap ]
Content:	Structure remains, the data just gets reduced to <1000 rows, while keeping an equal split.
```
<br>


## 6. Prediction and Signals

<div align="justify">
<p>  To get an overall view of the performance and accuracy of our model, I let the model assign different signals based on the predicted change. To keep the simulation simple, I stuck to BUY, HOLD & SELL; there is obviously big room for improvement and additional features, which could be tackled in future work. In order to choose the signal, we have to set or calculate a threshold. I implemented three different modes for that case:</p>


-  <b>STATIC:</b> Set a static percentage, e.g. 1%: the model would buy over a 1% predicted change and sell if < -1%.
- <b>PERCENTAGE:</b> Top/Bottom percentage for BUY/SELL, e.g. 10%: BUY Top 10% / SELL Bottom 10% of predicted change.
-  <b>NORMAL_DISTRIBUTION:</b> Only applies if the predicted change would be normally distributed, which has been disproven. Currently a relic of past versions, maybe relevant again in future work.
</div>

**Function:** 
```python
def simulate_predictions(model, df, mode, threshold):
    """
    Simulate stock signals based on calculated predicted changes.

    Args:
    - model: Pre-trained language model for prediction.
    - df (DataFrame): DataFrame containing news.
    - mode (str): Mode for threshold calculation ("static", "normal_distribution" or "percentage").
    - threshold (float): Threshold for categorization, either static value, percentile or percentage.

    Returns:
    - DataFrame: DataFrame containing simulated stock signals.
    """
    p_df = predict_df(model, df)  # predict the changes for each row

    if mode == "STATIC":
        p_df["signal"] = p_df["predicted change"].apply(categorize_prediction, pos_threshold=threshold, neg_threshold=(-threshold))
        print(f"Mode: {mode} - Positive: {threshold}, Negative: {-threshold}")
        plot_distribution(p_df, threshold, -threshold)

    elif mode == "NORMAL_DISTRIBUTION" or mode == "PERCENTAGE":
        pos_t, neg_t = calculate_auto_threshold(p_df["predicted change"], mode, threshold)
        p_df["signal"] = p_df["predicted change"].apply(categorize_prediction, pos_threshold=pos_t, neg_threshold=neg_t)
        print(f"Mode: {mode}, with thresholds for {threshold} percentile - Positive: {pos_t:.2f}, Negative: {neg_t:.2f}")
        plot_distribution(p_df, pos_t, neg_t) 

    else: 
        raise ValueError("Please use one of the following modes: STATIC, NORMAL_DISTRIBUTION or PERCENTAGE.")

    return p_df

def predict_df(model, df):
    """
    Predict stock price changes for a DataFrame by calling the prediction fun for each row.
	[...]
    """
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    prediction_result = df.apply(lambda row: predict_row(row, tokenizer, model), axis=1)
    predicted_df = pd.DataFrame(prediction_result.tolist())
    
    return predicted_df

def calculate_auto_threshold(predicted_changes, mode, percentile):
    """
    Calculate automatic thresholds based on the specified mode.
	[...]
    """
    if mode == "NORMAL_DISTRIBUTION":
        mean, std_dev = predicted_changes.mean(), predicted_changes.std()
    
        adj_percentile = percentile / 100

        top_threshold = stats.norm.ppf(1-adj_percentile, loc=mean, scale=std_dev)
        bottom_threshold = stats.norm.ppf(adj_percentile, loc=mean, scale=std_dev)

        return top_threshold, bottom_threshold

    elif mode == "PERCENTAGE":
        top_threshold = np.percentile(predicted_changes, 100-percentile)
        bottom_threshold = np.percentile(predicted_changes, percentile)

        return top_threshold, bottom_threshold

def categorize_prediction(prediction, pos_threshold, neg_threshold):
    """
    Categorize predicted changes into BUY, SELL, or HOLD based on a threshold.
	[...]
    """
    if prediction > pos_threshold: return "BUY"
    elif prediction < neg_threshold: return "SELL"
    else: return "HOLD"

def predict_row(row, tokenizer, model): [...]

def plot_distribution(df, pos_t, neg_t, predicted_col, actual_col): [...]
```

**Usage:** 
```python
# Params:
model 		= bert(model_name, train, val, model_type, learning_rate, epochs, batch_size, weight_decay) # as before in point 4
small_test	= sample_rows(test, sample_size, random_seed) # as before in point 5
mode 		= TMode.PERCENTAGE
threshold 	= 10

# Function Call:
sim 		= simulate_predictions(model, small_test, mode, threshold)
```

**Returns:** 
```
Type:		Pandas.DataFrame
Structure:	[ ticker | title | date | predicted change | future date | actual change | signal ]
Content:	Containing all predictions from the model for each news, next to the actual change.
```

**Prints:** 
```
Showing the (calculated) thresholds for the signal (BUY/HOLD/SELL) allocations.
2x MatPlotLib.Pyplot, displaying the distribution of the Predicted and Actual Changes
```
<br>


## 7. Performance Evaluation

<div align="justify">
<p>After I've added signals to each news, I continue with evaluating the predictions my model has made to get the mentioned overview of the performance and accuracy. We do that by calculating the average and median <b>actual</b> change of all BUY signals, hopefully with a positive result. The same is applied to all SELL signals, aiming for a negative value, since that is the prevented loss. We combine these two values for a final result, with some additional stats containing the best & worst predictions, next to the average error grouped by ticker.</p>
</div>

**Function:** 
```python
def evaluate_performance(sim_df, target, num_predictions):
    """
    Evaluate the performance of stock value predictions and their signal within a specified target date.

    Args:
    - sim_df (DataFrame): DataFrame containing simulated stock signals.
    - target (str): Target date identifier (e.g., "2W", "1M", "3M").
    - num_predictions (int): Number of top and bottom predictions to display.

    Returns:
    - best (DataFrame): DataFrame containing the best predictions.
    - worst (DataFrame): DataFrame containing the worst predictions.
    - combined_df (DataFrame): DataFrame containing count, sum, and average errors for each ticker.

    Prints:
    - Total change in stock value.
    - Win (+) or Loss (-) in mean and median percentage.
    """
    print(f"Calculations for a target in {target}")

    # get the value of all stocks we want to buy, today and in target date
    buy_signals = sim_df[sim_df["signal"] == "BUY"]
    total_change_buy_mean, total_change_buy_median = calculate_mm_change(buy_signals, "BUY")

    # get the value of all stocks we want to sell, today and in the future
    sell_signals = sim_df[sim_df["signal"] == "SELL"]
    total_change_sell_mean, total_change_sell_median = calculate_mm_change(sell_signals, "SELL")

    # check if it was a good decision to buy/sell the stocks
    total_change_mean = total_change_buy_mean + (-total_change_sell_mean)
    total_change_median = total_change_buy_median + (-total_change_sell_median)
    print(f"Win (+) or Loss (-) | Mean: {total_change_mean:.4f}%, Median: {total_change_median:.4f}%")

    # find the stocks with the best and worst predictions
    best, worst, avg_errors, sum_errors, count_of_tickers = find_best_worst_predictions(sim_df, num_predictions)

    count_of_tickers_df = count_of_tickers.to_frame(name="Count of Rows")
    sum_errors_df = sum_errors.to_frame(name="Sum Errors")
    avg_errors_df = avg_errors.to_frame(name="Average Errors")

    combined_df = (pd.concat([count_of_tickers_df, sum_errors_df, avg_errors_df], axis=1)).sort_values(by="Average Errors")
    
    return best, worst, combined_df

def calculate_mm_change(signals_df, signal_type):
    """
    Calculate the mean and median change based on stock signals.
    [...]
    """
    if signals_df.empty:
        return 0

    change_mean = signals_df["actual change"].mean()
    change_median = signals_df["actual change"].median()
    count_of_rows = signals_df.shape[0]

    print(f"Adjusted Change for {signal_type} | Mean: {change_mean:.4f}%, Median: {change_median:.4f}%, for a total amount of {count_of_rows} rows.")
    return change_mean, change_median 
    
def find_best_worst_predictions(df, num_predictions):
    """
    Find the best, worst predictions and some other stats from a DataFrame.
    [...]
    """
    df["difference"] = abs(df["predicted change"] - df["actual change"])
    sorted_df = df.sort_values("difference")

    best = sorted_df.head(num_predictions)

    # drop rows where the mathematical symbol is identical
    sorted_df = sorted_df[~(sorted_df["predicted change"] * sorted_df["actual change"] > 0)]

    worst = sorted_df.tail(num_predictions)

    # calculate average and sum of errors for each ticker
    avg_errors = df.groupby("ticker")["difference"].mean()
    sum_errors = df.groupby("ticker")["difference"].sum()
    count_of_tickers = df.groupby("ticker").size()

    return best, worst, avg_errors, sum_errors, count_of_tickers
```

**Usage:** 
```python
# Params:
sim 		= simulate_predictions(model, small_test, mode, threshold) # as before in point 6
target 		= Target.W1
num_pred 	= 100

# Function Call:
best, worst, combined_df = evaluate_performance(sim, target, num_pred)
```

**Returns:** 
```
Type:		3x Pandas.DataFrame 

Best, Worst:
Structure:	[ ticker | title | date | predicted change | future date | actual change | signal | difference ]
Content:	The 100 (num_pred) best & worst predictions based on the difference between actual and predicted change.

Combined_df:
Structure:	[ ticker | Count of Rows | Sum Errors | Average Errors ]
Content:	Stats about the amount of news-articles per ticker in the testdata, sorted by average error.
```

**Prints:** 
```
Displays the average and mean (actual) change for the BUY and SELL signal, with the potential profit or loss.
```
<br>
