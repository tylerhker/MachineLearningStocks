import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from utils import data_string_to_float, status_calc, status_calc_self
from download_historical_prices import build_result_dataset


# The percentage by which a stock has to beat the S&P500 to be considered a 'buy'
OUTPERFORMANCE = 10


def build_data_set():
    """
    Reads the keystats.csv file and prepares it for scikit-learn
    :return: X_train and y_train numpy arrays
    """
    training_data = pd.read_csv("keystats.csv", index_col="Date")
    training_data.drop([
                # "Market Cap",
                "Enterprise Value",
                # "Trailing P/E",
                # "Forward P/E",
                # "PEG Ratio",
                # "Price/Sales",
                # "Price/Book",
                # "Enterprise Value/Revenue",
                # "Enterprise Value/EBITDA",
                # "Profit Margin",
                # "Operating Margin",
                # "Return on Assets",
                # "Return on Equity",
                # "Revenue",
                # "Revenue Per Share",
                "Qtrly Revenue Growth",
                # "Gross Profit",
                # "EBITDA",
                "Net Income Avl to Common",
                # "Diluted EPS",
                "Qtrly Earnings Growth",
                # "Total Cash",
                # "Total Cash Per Share",
                "Total Debt",
                # "Total Debt/Equity",
                # "Current Ratio",
                # "Book Value Per Share",
                "Operating Cash Flow",
                "Levered Free Cash Flow",
                # "Beta",
                # "50-Day Moving Average",
                # "200-Day Moving Average",
                # "Avg Vol (3 month)",
                # "Shares Outstanding",
                # "Float",
                # "% Held by Insiders",
                # "% Held by Institutions",
                # "Shares Short (as of",
                # "Short Ratio",
                # "Short % of Float",
                # "Shares Short (prior month)",
                ], axis=1,inplace=True)
    training_data.dropna(axis=0, how="any", inplace=True)
    features = training_data.columns[6:]

    X_train = training_data[features].values
    # Generate the labels: '1' if a stock beats the S&P500 by more than 10%, else '0'.
    # y_train = list(
    #     status_calc(
    #         training_data["stock_p_change"],
    #         training_data["SP500_p_change"],
    #         OUTPERFORMANCE,
    #     )
    # )

    #Generate the labels: '1' if the stock price is increased by more than X%, else '0'.
    y_train = list(
       status_calc_self(
           training_data["stock_p_change"], outperformance=15
           )
       )
    return X_train, y_train


def predict_stocks():
    X_train, y_train = build_data_set()
    # Remove the random_state parameter to generate actual predictions
    # clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf = RandomForestClassifier(
                                bootstrap= False,
                                max_depth=None,
                                max_features= 'sqrt',
                                min_samples_leaf= 1,
                                min_samples_split= 3,
                                n_estimators= 200
                                )
    
    clf.fit(X_train, y_train)

    # Now we get the actual data from which we want to generate predictions.
    data = pd.read_csv("forward_sample.csv", index_col="Date")
    data.drop([
                # "Market Cap",
                "Enterprise Value",
                # "Trailing P/E",
                # "Forward P/E",
                # "PEG Ratio",
                # "Price/Sales",
                # "Price/Book",
                # "Enterprise Value/Revenue",
                # "Enterprise Value/EBITDA",
                # "Profit Margin",
                # "Operating Margin",
                # "Return on Assets",
                # "Return on Equity",
                # "Revenue",
                # "Revenue Per Share",
                "Quarterly Revenue Growth",
                # "Gross Profit",
                # "EBITDA",
                "Net Income Avi to Common",
                # "Diluted EPS",
                "Quarterly Earnings Growth",
                # "Total Cash",
                # "Total Cash Per Share",
                "Total Debt",
                # "Total Debt/Equity",
                # "Current Ratio",
                # "Book Value Per Share",
                "Operating Cash Flow",
                "Levered Free Cash Flow",
                # "Beta",
                # "50-Day Moving Average",
                # "200-Day Moving Average",
                # "Avg Vol (3 month)",
                # "Shares Outstanding",
                # "Float",
                # "% Held by Insiders",
                # "% Held by Institutions",
                # "Shares Short (as of",
                # "Short Ratio",
                # "Short % of Float",
                # "Shares Short (prior month)",
                ], axis=1,inplace=True)
    
    data.dropna(axis=0, how="any", inplace=True)
    features = data.columns[6:]
    X_test = data[features].values
    z = data["Ticker"].values

    # Get the predicted tickers
    y_pred = clf.predict(X_test)
    if sum(y_pred) == 0:
        print("No stocks predicted!")
    else:
        invest_list = z[y_pred].tolist()
        print(
            f"{len(invest_list)} stocks predicted to outperform the S&P500 by more than {OUTPERFORMANCE}%:"
        )
        print(" ".join(invest_list))
        return invest_list

def list_ticker_result():
    with open('result.txt','r') as f:
        ticker_result = csv.reader(f,delimiter=' ')        
        tickersall=[]
        for ticker in ticker_result:
            tickersall.append(ticker)

        tickersall = tickersall[0]

    return tickersall

if __name__ == "__main__":
    print("Building dataset and predicting stocks...")
    predict_stocks()
    ticker_list = list_ticker_result()
    build_result_dataset(ticker_list, start='2021-01-01', end='2023-01-31')
