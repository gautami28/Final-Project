from flask import Blueprint, request, jsonify

chatbot_bp = Blueprint('chatbot_bp', __name__)

# Hardcoded stock prices for demonstration
STOCK_PRICES = {
    'AAPL': 192.5,
    'MSFT': 332.8,
    'NVDA': 480.6,
    'TSLA': 288.2,
    'AMZN': 145.1,
    'META': 198.5,
    'NFLX': 515.3,
    'INTC': 50.2,
    'ORCL': 88.3,
    'CSCO': 55.1,
}

@chatbot_bp.route('/chatbot', methods=['POST'])
def chatbot():
    user_message = request.json.get('message', '').strip().lower()
    response = "Sorry, I can only answer questions about stock prices."

    # Small talk & greetings
    if any(greet in user_message for greet in ['hello', 'hi', 'hey']):
        response = "Hello! How can I help you with stocks today?"
    elif "thank" in user_message:
        response = "You're welcome! Let me know if you have more questions."
    elif "bye" in user_message or "goodbye" in user_message:
        response = "Goodbye! Have a great day trading!"

    # FAQ
    elif "what is a stock" in user_message:
        response = "A stock is a type of security that gives you ownership in a company."
    elif "how do i use" in user_message or "dashboard" in user_message:
        response = "To use the dashboard, select a stock, enter the price and news, and click Predict."
    elif "sentiment analysis" in user_message:
        response = "Sentiment analysis measures the positivity or negativity of news headlines about a stock."
    elif "how do i get a prediction" in user_message or "predict" in user_message:
        response = "To get a prediction, select a stock, enter the closing price and news, then click Predict."

    # Stock price
    else:
        for ticker in STOCK_PRICES:
            if ticker.lower() in user_message:
                response = f"The current price of {ticker} is ${STOCK_PRICES[ticker]:.2f}."
                break
        if 'price' in user_message and response.startswith('Sorry'):
            response = "I don't have data for that ticker. Try AAPL, MSFT, NVDA, TSLA, AMZN, META, NFLX, INTC, ORCL, or CSCO."

    return jsonify({'response': response}) 