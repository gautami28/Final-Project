# Stock Prediction AI Application

A comprehensive Flask web application that uses machine learning models to predict stock prices and analyze market sentiment.

## Features

- **Stock Price Prediction**: AI-powered predictions using Bidirectional GRU models
- **Sentiment Analysis**: News sentiment classification for market insights
- **Interactive Dashboard**: Real-time market data visualization
- **AI Chatbot**: Intelligent assistant for stock-related queries
- **Combined Analysis**: Price and sentiment correlation analysis

## Supported Stocks

The application supports predictions for 50+ major stocks including:
- AAPL (Apple Inc.)
- MSFT (Microsoft Corporation)
- NVDA (NVIDIA Corporation)
- TSLA (Tesla Inc.)
- AMZN (Amazon.com Inc.)
- META (Meta Platforms Inc.)
- GOOGL (Alphabet Inc.)
- And many more...

## Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Machine Learning**: TensorFlow/Keras, Bidirectional GRU models
- **Data Visualization**: Plotly
- **Data Processing**: NumPy, Pandas, Scikit-learn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/gautami28/Final-Project.git
cd Final-Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Project Structure

```
model_artifacts/
├── app.py                          # Main Flask application
├── chatbot.py                      # AI chatbot implementation
├── train_model.py                  # Model training script
├── requirements.txt                # Python dependencies
├── new_UI/                         # Frontend assets
│   ├── templates/                  # HTML templates
│   ├── static/                     # CSS, JS, images
│   └── models/                     # Trained ML models
└── model_artifacts/                # Original model artifacts
```

## Usage

### Dashboard
- View market overview with top gainers and losers
- Monitor AI predictions vs actual prices
- Access quick metrics and portfolio information

### Stock Prediction
1. Select a stock ticker from the dropdown
2. Enter current stock price
3. Add relevant news text (optional)
4. Get AI-powered price predictions

### Sentiment Analysis
- Analyze news sentiment for market insights
- Classify text as bullish, bearish, or neutral

### AI Chatbot
- Ask questions about stocks, predictions, or market trends
- Get intelligent responses powered by AI

## Model Information

- **Architecture**: Bidirectional GRU with enhanced sentiment analysis
- **Input Features**: Price data and news sentiment
- **Output**: Predicted stock prices
- **Accuracy**: ~88.8% (varies by stock)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This application is for educational and research purposes only. Stock predictions are not financial advice. Always consult with a financial advisor before making investment decisions.

## Contact

For questions or support, please open an issue on GitHub.
