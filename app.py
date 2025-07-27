import os
import pickle
from flask import Flask, render_template, request, jsonify
import plotly.graph_objs as go
import plotly
import json
import numpy as np
import tensorflow as tf  # <-- Added for Keras model loading
import joblib
from keras.models import load_model
from chatbot import chatbot_bp

app = Flask(__name__, template_folder='new_UI/templates', static_folder='new_UI/static')
app.register_blueprint(chatbot_bp)

# Path to models folder
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'new_UI', 'models')

def get_available_tickers():
    tickers = set()
    if not os.path.exists(MODELS_DIR):
        print(f"[ERROR] MODELS_DIR not found: {MODELS_DIR}")
        return []
    for fname in os.listdir(MODELS_DIR):
        if fname.endswith("_feat.pkl"):
            ticker = fname.split("_")[0].upper()
            tickers.add(ticker)
    return sorted(tickers)


VALID_TICKERS = get_available_tickers()

# Global dummy historical data for demonstration
history_prices = [188.5, 189.7, 190.1, 191.2, 192.5]
history_dates = ["Mon", "Tue", "Wed", "Thu", "Fri"]

@app.route('/')
def dashboard():
    gainers = [
        {'ticker': 'AAPL', 'name': 'Apple Inc.', 'change': '+3.2%'},
        {'ticker': 'MSFT', 'name': 'Microsoft Corporation', 'change': '+2.8%'},
        {'ticker': 'NVDA', 'name': 'NVIDIA Corporation', 'change': '+2.4%'},
        {'ticker': 'TSLA', 'name': 'Tesla Inc.', 'change': '+2.1%'},
        {'ticker': 'AMZN', 'name': 'Amazon.com Inc.', 'change': '+1.9%'},
    ]
    losers = [
        {'ticker': 'META', 'name': 'Meta Platforms Inc.', 'change': '-2.3%'},
        {'ticker': 'NFLX', 'name': 'Netflix Inc.', 'change': '-2.2%'},
        {'ticker': 'INTC', 'name': 'Intel Corporation', 'change': '-2.1%'},
        {'ticker': 'AMD', 'name': 'Advanced Micro Devices', 'change': '-1.8%'},
        {'ticker': 'GOOGL', 'name': 'Alphabet Inc.', 'change': '-1.6%'},
    ]
    metrics = [
        {'icon':'bi-graph-up', 'label':'Model Accuracy', 'value':'88.8%'},
        {'icon':'bi-wallet2', 'label':'Portfolio Value', 'value':'$120.5K'},
        {'icon':'bi-clock-history', 'label':'Last Update', 'value':'few mins ago'},
        {'icon':'bi-people', 'label':'Active Users', 'value':'1'},
    ]
    predictions = [
        {'ticker': 'AAPL', 'price': 195.32, 'actual': 192.5, 'percent': 1.47},
        {'ticker': 'MSFT', 'price': 335.45, 'actual': 332.8, 'percent': 0.80},
        {'ticker': 'TSLA', 'price': 290.76, 'actual': 288.2, 'percent': 0.89},
        {'ticker': 'AMZN', 'price': 148.99, 'actual': 145.1, 'percent': 2.68},
    ]
    stock_list = [
        {'ticker': 'AAPL', 'name': 'Apple Inc.', 'price': '192.5'},
        {'ticker': 'MSFT', 'name': 'Microsoft Corporation', 'price': '332.8'},
        {'ticker': 'NVDA', 'name': 'NVIDIA Corporation', 'price': '480.6'},
        {'ticker': 'TSLA', 'name': 'Tesla Inc.', 'price': '288.2'},
        {'ticker': 'AMZN', 'name': 'Amazon.com Inc.', 'price': '145.1'},
        {'ticker': 'META', 'name': 'Meta Platforms Inc.', 'price': '198.5'},
        {'ticker': 'NFLX', 'name': 'Netflix Inc.', 'price': '515.3'},
        {'ticker': 'INTC', 'name': 'Intel Corporation', 'price': '50.2'},
        {'ticker': 'GOOGL', 'name': 'Alphabet Inc.', 'price': '142.8'},
        {'ticker': 'AMD', 'name': 'Advanced Micro Devices', 'price': '125.4'},
    ]
    watchlist = [
        {'ticker': 'AAPL', 'name': 'Apple Inc.', 'price': '192.5'},
        {'ticker': 'TSLA', 'name': 'Tesla Inc.', 'price': '288.2'},
        {'ticker': 'NVDA', 'name': 'NVIDIA Corporation', 'price': '480.6'},
    ]
    return render_template(
        'dashboard.html',
        gainers=gainers,
        losers=losers,
        metrics=metrics,
        predictions=predictions,
        stock_list=stock_list,
        watchlist=watchlist
    )


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    tickers = get_available_tickers()
    prediction = None
    plot_json = None
    error = None

    if request.method == 'POST':
        ticker = request.form.get('ticker', '').strip().upper()
        price = request.form.get('price')
        news = request.form.get('news', '')

        if ticker not in VALID_TICKERS:
            error = f"Ticker '{ticker}' is not supported."
        else:
            feat_path = os.path.join(MODELS_DIR, f"{ticker}_feat.pkl")
            targ_path = os.path.join(MODELS_DIR, f"{ticker}_targ.pkl")
            model_path = os.path.join(MODELS_DIR, f"{ticker}_gru.keras")

            if not os.path.exists(feat_path):
                error = f"Feature scaler for '{ticker}' not found."
            elif not os.path.exists(model_path):
                error = f"Model file for '{ticker}' not found."
            else:
                try:
                    with open(os.path.join(MODELS_DIR, f"{ticker}_feat.pkl"), "rb") as f:
                        feat_scaler = joblib.load(f)

                    targ_scaler = None
                    if os.path.exists(targ_path):
                        with open(os.path.join(MODELS_DIR, f"{ticker}_targ.pkl"), "rb") as f:
                            targ_scaler = joblib.load(f)

                    # Create a sequence of 30 time steps, each with the current input
                    X_orig = np.array([[float(price), len(news)]] * 30)  # shape (30, 2)
                    X_scaled = feat_scaler.transform(X_orig)  # shape (30, 2)
                    X_scaled = X_scaled.reshape(1, 30, 2)    # shape (1, 30, 2) for batch size 1

                    model = tf.keras.models.load_model(model_path)
                    predicted_scaled = model.predict(X_scaled)

                    if targ_scaler:
                        predicted = targ_scaler.inverse_transform(predicted_scaled)[0][0]
                    else:
                        predicted = predicted_scaled[0][0]

                    predicted_price = round(float(predicted), 2)

                    prediction = {
                        'ticker': ticker,
                        'price': price,
                        'news': news,
                        'predicted_price': predicted_price,
                    }

                    prices_plot = history_prices + [predicted_price]
                    dates_plot = history_dates + ["Prediction"]

                    fig = go.Figure([
                        go.Scatter(x=dates_plot, y=prices_plot, mode='lines+markers', name='Price')
                    ])
                    fig.add_trace(go.Scatter(
                        x=[dates_plot[-1]], y=[prices_plot[-1]],
                        mode='markers+text',
                        marker=dict(color='orange', size=14),
                        name='Predicted',
                        text=['Predicted'],
                        textposition='top center'
                    ))
                    fig.update_layout(
                        title=f"{ticker} Price Prediction",
                        yaxis_title="Price ($)",
                        template="simple_white",
                        margin=dict(l=20, r=20, t=40, b=20),
                        height=340
                    )
                    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

                except Exception as e:
                    error = f"Prediction failed: {str(e)}"

    return render_template('predict.html', tickers=VALID_TICKERS, prediction=prediction, plot_json=plot_json, error=error)



@app.route('/analyze')
def analyze():
    prices = [110, 112, 115, 117, 121, 119, 123]
    sentiment = [0.2, 0.35, 0.1, 0.6, 0.8, 0.3, 0.65]
    dates = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=prices, name='Price', yaxis="y1", mode='lines+markers'))
    fig.add_trace(go.Bar(x=dates, y=sentiment, name='Sentiment', yaxis="y2", opacity=0.4))
    fig.update_layout(
        title="Combined Price & News Sentiment",
        yaxis=dict(title="Price", side='left'),
        yaxis2=dict(title="Sentiment", overlaying='y', side='right', range=[0, 1]),
        legend=dict(x=0, y=1.1, orientation="h"),
        template="simple_white",
        margin=dict(l=20, r=20, t=40, b=20),
        height=340
    )
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('analyze.html', plot_json=plot_json)


@app.route('/sentiment')
def sentiment():
    return render_template('sentiment.html')


@app.route('/chat')
def chat():
    return render_template('chat.html')


@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    text = request.json.get('text', '').lower()
    positive_words = ['increase', 'up', 'gain', 'rise', 'surge', 'positive', 'profit']
    negative_words = ['decrease', 'down', 'loss', 'drop', 'fall', 'negative', 'decline']

    score = 0
    for word in positive_words:
        if word in text:
            score += 1
    for word in negative_words:
        if word in text:
            score -= 1

    if score > 0:
        sentiment = 'Positive'
    elif score < 0:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    return jsonify({'sentiment': sentiment, 'score': score})


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
