import os
import random
import numpy as np
import pandas as pd
import torch
import tensorflow as tf
from tqdm.auto import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
import joblib

# ─── Reproducibility ─────────────────────────────────────────────────────
import random
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)

# ─── Configuration ───────────────────────────────────────────────────────
SEQUENCE_LENGTH = 30
BATCH_SIZE = 32
MAX_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 20
FINBERT_BATCH_SIZE = 64

MODEL_SAVE_DIR  = 'stock_prediction_models_bidirectional_gru_enhanced_sentiment_reanalyzed'
SCALER_SAVE_DIR = 'scalers_bidirectional_gru_enhanced_sentiment_reanalyzed'
MERGED_DATA_SAVE_FILE = 'merged_stock_sentiment_data_reanalyzed.csv'

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(SCALER_SAVE_DIR, exist_ok=True)

# ─── 1. Load & Prep Data ─────────────────────────────────────────────────
print("\n--- 1. Data Loading and Preprocessing ---")
try:
    news_df  = pd.read_csv('aligned_news_data.csv',  parse_dates=['Date'], dayfirst=True)
    stock_df = pd.read_csv('aligned_stock_data.csv', parse_dates=['Date'])
except FileNotFoundError:
    print("Missing CSVs in working dir. Exiting.")
    exit()

news_df  = news_df.sort_values('Date').reset_index(drop=True)
stock_df = stock_df.sort_values('Date').reset_index(drop=True)

# ─── 2. FinBERT Sentiment ─────────────────────────────────────────────────
print("\n--- 2. Optimized Sentiment Analysis with FinBERT ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer     = AutoTokenizer.from_pretrained("ProsusAI/finbert")
finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
finbert_model.eval()

def get_finbert_probs(texts, batch_size=FINBERT_BATCH_SIZE):
    out = []
    cleaned = [str(t) if pd.notna(t) else "" for t in texts]
    for i in tqdm(range(0, len(cleaned), batch_size), desc="FinBERT batches"):
        batch = cleaned[i:i+batch_size]
        nonempty = [t for t in batch if t]
        if not nonempty:
            out.extend([[0.,0.,0.]] * len(batch))
            continue
        inputs = tokenizer(nonempty, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = finbert_model(**inputs).logits
        probs = softmax(logits, dim=1).cpu().numpy()
        idx = 0
        for t in batch:
            if not t:
                out.append([0.,0.,0.])
            else:
                out.append(probs[idx].tolist())
                idx += 1
    return np.array(out)

# compute & save refreshed CSV so you can skip re-running this each time
fin_probs = get_finbert_probs(news_df['Headline'].tolist())
news_df['finbert_neutral_prob']  = fin_probs[:,0]
news_df['finbert_positive_prob'] = fin_probs[:,1]
news_df['finbert_negative_prob'] = fin_probs[:,2]
news_df['finbert_score']         = news_df['finbert_positive_prob'] - news_df['finbert_negative_prob']
news_df['finbert_strength']      = np.maximum(news_df['finbert_positive_prob'], news_df['finbert_negative_prob'])

news_df.to_csv('news_with_finbert.csv', index=False)
stock_df.to_csv('stock_raw.csv',         index=False)

# ─── 3. Merge with Shifted Sentiment ──────────────────────────────────────
print("\n--- 3. Merging Stock & Shifted Sentiment ---")
daily = (
    news_df
      .groupby(['Date','Symbol'])
      .finbert_score.mean()
      .reset_index(name='mean_score')
)
daily['Date'] += pd.Timedelta(days=1)
merged = pd.merge(stock_df, daily, on=['Date','Symbol'], how='left').fillna(0)
merged.to_csv(MERGED_DATA_SAVE_FILE, index=False)

# ─── 4. Feature Engineering ──────────────────────────────────────────────
print("\n--- 4. Enhanced Feature Engineering ---")
def create_features(df):
    df = df.copy()
    df['P1'] = df.groupby('Symbol')['Close'].shift(1)
    df['P2'] = df.groupby('Symbol')['Close'].shift(2)
    df['P3'] = df.groupby('Symbol')['Close'].shift(3)
    df['Return'] = (df['Close'] - df['P1']) / df['P1']
    df['Vol_Chg'] = (df['Volume'] - df.groupby('Symbol')['Volume'].shift(1)) / df.groupby('Symbol')['Volume'].shift(1)
    df['HighLow'] = (df['High'] - df['Low']) / df['Open']
    for w in (3,5,7):
        df[f'roll_mean_{w}'] = df.groupby('Symbol')['mean_score'].transform(lambda x: x.rolling(w,1).mean())
        df[f'roll_std_{w}']  = df.groupby('Symbol')['mean_score'].transform(lambda x: x.rolling(w,1).std()).fillna(0)
    return df.fillna(0)

data = create_features(merged)

# ─── 5. Train per Symbol ──────────────────────────────────────────────────
print("\n--- 5. Training Models Per Stock ---")
symbols = data['Symbol'].unique()
perf    = []

for sym in symbols:
    print(f"\n> {sym}")
    sub = data[data['Symbol'] == sym].dropna().copy()
    if len(sub) < SEQUENCE_LENGTH + 1:
        print(" skip: not enough rows")
        continue

    # Use only Close and mean_score
    sub['Next_Close'] = sub['Close'].shift(-1)
    sub['Delta_Close'] = sub['Next_Close'] - sub['Close']
    sub.dropna(subset=['Delta_Close'], inplace=True)

    features = ['Close', 'mean_score']
    target = 'Delta_Close'

    feature_scaler = MinMaxScaler()
    X_scaled = feature_scaler.fit_transform(sub[features])
    y = sub[target].values.reshape(-1, 1)

    # Save the feature scaler
    joblib.dump(feature_scaler, os.path.join(SCALER_SAVE_DIR, f'{sym}_feat.pkl'))

    # Create sequences
    def create_sequences(X, y, seq_len):
        Xs, ys = [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i:i + seq_len])
            ys.append(y[i + seq_len][0])
        return np.array(Xs), np.array(ys)

    X_seq, y_seq = create_sequences(X_scaled, y, SEQUENCE_LENGTH)
    if X_seq.shape[0] < 10:
        print(" skip: too few sequences")
        continue

    # Train/test split
    split = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    tf.keras.backend.clear_session()
    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True),
                                      input_shape=(SEQUENCE_LENGTH, len(features))),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=20, batch_size=32,
              validation_split=0.1, callbacks=[early_stop], verbose=0)

    # Save model
    model_path = os.path.join(MODEL_SAVE_DIR, f'{sym}_gru.keras')
    model.save(model_path)
    print(" model saved to", model_path)

    # Directional accuracy metric
    pred = model.predict(X_test)
    true_dir = np.sign(y_test)
    pred_dir = np.sign(pred.flatten())
    dir_acc = (true_dir == pred_dir).mean()
    print(f" Directional Accuracy: {dir_acc:.2f}")

    # clear session
    tf.keras.backend.clear_session()

# ─── 6. Summary ─────────────────────────────────────────────────────────
perf_df = pd.DataFrame(perf)
print("\nOverall performance:\n", perf_df)