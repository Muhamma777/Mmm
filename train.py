import pandas as pd
import joblib
import argparse
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss
from lightgbm import LGBMClassifier
from features import build_features_for_match

# Simple training script for Over-total classifier (binary: Over thresh -> 1)

def build_dataset(df, threshold=2.5):
    rows = []
    for idx, row in df.sort_values('date').iterrows():
        home = row['home_team']
        away = row['away_team']
        date = row['date']
        total_goals = int(row['home_goals']) + int(row['away_goals'])
        label = 1 if total_goals > threshold else 0
        feats = build_features_for_match(home, away, date, threshold)
        if feats is None:
            continue
        feats['label'] = label
        rows.append(feats)
    if not rows:
        return None, None
    df_feats = pd.DataFrame(rows)
    X = df_feats.drop(columns=['label'])
    y = df_feats['label'].astype(int)
    return X, y

def main():
    parser = argparse.ArgumentParser(description='Train Over-total model')
    parser.add_argument('--data', default='data/matches.csv', help='Path to matches CSV')
    parser.add_argument('--out', default='model.joblib', help='Output model file')
    parser.add_argument('--threshold', type=float, default=2.5, help='Total threshold (2.5 or 1.5)')
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Data file not found: {args.data}\nYou can use the sample data data/matches_sample.csv as a starting point.")
        return

    df = pd.read_csv(args.data, parse_dates=['date'])
    X, y = build_dataset(df, threshold=args.threshold)
    if X is None or y is None or X.shape[0] == 0:
        print('Not enough training data after feature extraction. Need more historical matches.')
        return

    # scaler
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # simple time-series split for evaluation
    tscv = TimeSeriesSplit(n_splits=3)
    accs = []
    losses = []
    for train_idx, test_idx in tscv.split(Xs):
        X_train, X_test = Xs[train_idx], Xs[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = LGBMClassifier(n_estimators=200, learning_rate=0.05)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)
        accs.append(accuracy_score(y_test, preds))
        try:
            losses.append(log_loss(y_test, probs))
        except Exception:
            losses.append(None)

    print('CV accuracy:', accs)
    print('CV logloss:', losses)

    # final train on all data
    final_model = LGBMClassifier(n_estimators=500, learning_rate=0.03)
    final_model.fit(Xs, y)

    bundle = {'model': final_model, 'feature_order': list(X.columns), 'scaler': scaler}
    joblib.dump(bundle, args.out)
    print(f'Model saved to {args.out}')

if __name__ == '__main__':
    main()