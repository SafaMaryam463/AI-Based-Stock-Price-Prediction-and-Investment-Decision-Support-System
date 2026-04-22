from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np

# ---------------- INIT ----------------
app = Flask(__name__)

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- ADVICE + EXPLANATION ----------------
def get_advice(pred, current):
    change_percent = ((pred - current) / current) * 100

    if change_percent > 2:
        return "BUY 📈", f"Price expected to rise by {change_percent:.2f}%. Uptrend detected."
    elif change_percent < -2:
        return "SELL 📉", f"Price expected to drop by {abs(change_percent):.2f}%. Downtrend detected."
    else:
        return "HOLD 🤝", f"Price change is small ({change_percent:.2f}%). Market is stable."

# ---------------- HOME ----------------
@app.route('/')
def home():
    return render_template("index.html")

# ---------------- PREDICT ----------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        stock = request.form['stock']
        investment = float(request.form['investment'])
        days = int(request.form['days'])

        # ---------------- LOAD DATA ----------------
        df = pd.read_csv(f"stocks/{stock}.csv")

        data = df[['Close']].values
        scaled_data = scaler.transform(data)

        # ---------------- SIMPLE MULTI-DAY PREDICTION ----------------
        future_prices = []

        last_value = scaled_data[-1][0]

        for _ in range(days):
            x_input = [[last_value]]
            pred = model.predict(x_input)[0]   # correct for LinearRegression

            # keep prediction stable (avoid crazy values)
            pred = max(0, min(pred, 1))

            future_prices.append([pred])
            last_value = pred

        # convert back to real price
        future_prices = scaler.inverse_transform(future_prices)

        # ---------------- CURRENT + FUTURE ----------------
        current_price = df['Close'].iloc[-1]
        predicted_price = future_prices[-1][0]

        # ---------------- PROFIT ----------------
        profit = ((predicted_price - current_price) / current_price) * investment

        # ---------------- ADVICE ----------------
        advice, reason = get_advice(predicted_price, current_price)

        # ---------------- FORMAT OUTPUT ----------------
        current_price = "{:,.2f}".format(current_price)
        predicted_price = "{:,.2f}".format(predicted_price)
        profit = "{:,.2f}".format(profit)

        # ---------------- RETURN ----------------
        return render_template("index.html",
                               stock=stock.upper(),
                               last_price=current_price,
                               prediction=predicted_price,
                               profit=profit,
                               days=days,
                               advice=advice,
                               reason=reason)

    except Exception as e:
        return f"Error: {str(e)}"

# ---------------- RUN ----------------
if __name__ == '__main__':
    app.run(debug=True)