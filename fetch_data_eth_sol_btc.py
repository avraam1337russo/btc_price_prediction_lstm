



import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from dotenv import load_dotenv

# --- НАСТРОЙКИ ---
load_dotenv()
API_KEY = os.getenv('KEY')
SYMBOL = "BTCUSDT"
EXCHANGE = "Bybit"
INTERVAL = "1d"

HEADERS = {"accept": "application/json", "CG-API-KEY": API_KEY}


def fetch_coinglass(endpoint, params):
    url = f"https://open-api-v4.coinglass.com{endpoint}"
    try:
        response = requests.get(url, params=params, headers=HEADERS)
        res = response.json()
        if res.get("code") == "0" and res.get("data"):
            data = res["data"]
            # В некоторых ответах данные лежат в списке напрямую, в других в ключе 'list'
            return pd.DataFrame(data if isinstance(data, list) else data.get("list", []))
    except Exception as e:
        print(f"Ошибка при запросе {endpoint}: {e}")
    return pd.DataFrame()

