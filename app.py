# app.py
from src.predict_utils import run_3class_prediction, run_2class_prediction

if __name__ == '__main__':
    ticker = input("Enter the IT stock ticker: ").upper()
    if ticker:
        run_3class_prediction(ticker)
        run_2class_prediction(ticker)