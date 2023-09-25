import requests

symbol = "AAPL" #Stock Symbol of Apple Inc.

#API key
api_key = "ABG95G8I3JFTXMH0"

#Time period for the Data to be fetched
slices = ["year1month1", "year1month2", "year1month3", "year1month4", "year1month5", "year1month6", "year1month7", "year1month8", "year1month9", "year1month10", "year1month11", "year1month12"]

#Fetching the data
for slice in slices:
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={symbol}&interval=5min&slice={slice}&apikey={api_key}"

    response = requests.get(url)

    with open(f"new_intraday_data.csv", "w") as file:
        file.write(response.text)
        print("Data saved to file")
