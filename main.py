# Save this as main.py
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd

# Load the saved model
loaded_model = joblib.load('volatility_predictor.pkl')
app = FastAPI()

# Get the feature names the model expects
required_features = loaded_model.feature_names_in_

# This dictionary maps the new app-friendly names to the original names
# you can generate this automatically or list it manually
feature_mapping = {
    # Features without spaces or special characters
    "Open": "Open",
    "High": "High",
    "Low": "Low",
    "Close": "Close",
    "Volume": "Volume",
    "SP500": "SP500",
    "VIX": "VIX",
    "DJIA": "DJIA",
    "NASDAQ": "NASDAQ",
    "RUSSELL": "RUSSELL",
    "Gold": "Gold",
    "Oil": "Oil",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    
    # Features with spaces or special characters
    "AdjClose": "Adj Close",
    "SP500_LogReturns": "SP500 Log Returns",
    "SP500_30DayVolatility": "SP500 30 Day Volatility",
    "SPX_PutCallRatio": "SPX Put Call Ratio",
    "SPX_PutVolume": "SPX Put Volume",
    "SPX_CallVolume": "SPX Call Volume",
    "Total_SPX_OptionsVolume": "Total SPX Options Volume",
    "10Y_Treasury": "10Y_Treasury",
    "High_Yield_Bonds": "High_Yield_Bonds",
    "EMB_Yield": "EMB_Yield",
    "MSCI_World": "MSCI_World",
    "Consumer_Sentiment": "Consumer_Sentiment",
    "USD_Index": "USD_Index",
    "DJIA_log_returns": "DJIA_log_returns",
    "NASDAQ_log_returns": "NASDAQ_log_returns",
    "RUSSELL_log_returns": "RUSSELL_log_returns",
    "MSCI_World_log_returns": "MSCI_World_log_returns",
    "USD_Index_log_returns": "USD_Index_log_returns",
    "Gold_log_returns": "Gold_log_returns",
    "Oil_log_returns": "Oil_log_returns",
    "SP500_put_log_change": "SP500_put_log_change",
    "SP500_call_log_change": "SP500_call_log_change",
    "SP500_total_opts_log_change": "SP500_total_opts_log_change",
    "Consumer_Sentiment_log_change": "Consumer_Sentiment_log_change"
}

@app.post("/predict_volatility")
def predict_volatility(data: dict):
    # Create a new dictionary to hold the corrected feature names
    processed_data = {}
    for key, value in data.items():
        # Use the mapping to get the original feature name
        original_key = feature_mapping.get(key, key)
        processed_data[original_key] = value

    # Create a DataFrame in the correct column order
    try:
        data_df = pd.DataFrame([processed_data], columns=required_features)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input data: {e}")

    # Make the prediction
    prediction = loaded_model.predict(data_df)

    # Return the prediction result as a JSON object
    return {"prediction": prediction[0].item()}