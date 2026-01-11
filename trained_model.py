import torch
import pandas as pd
import numpy as np
import joblib

from coinglass_model import LSTMModel

# 1. Загрузка компонентов
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ВАЖНО: Определите параметры точно так же, как при обучении
INPUT_DIM = 43  # Проверьте, что в features_list именно столько элементов
HIDDEN_DIM = 64
NUM_LAYERS = 2
OUTPUT_DIM = 1

# Инициализируем архитектуру и загружаем веса
model = LSTMModel(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM).to(device)
model.load_state_dict(torch.load('lstm_multi_xchange_data.pth', map_location=device))
model.eval()

# Загружаем скалеры
scaler_x = joblib.load('scaler_x.pkl')
scaler_y = joblib.load('scaler_y.pkl')
features_list = [
    'target_pct',  # (как история прошлых изменений)

    'bybit_btc_oi_change_pct',
    'bybit_btc_vol_change_pct',
    'bybit_BTCUSDT_ls_ratio',
    'bybit_BTCUSDT_taker_delta',
    'bybit_BTCUSDT_long_liquidation_usd',
    'bybit_BTCUSDT_short_liquidation_usd',
    'bybit_BTCUSDT_liq_delta',

    'bybit_eth_oi_change_pct',
    'bybit_eth_vol_change_pct',
    'bybit_ETHUSDT_ls_ratio',
    'bybit_ETHUSDT_taker_delta',
    'bybit_ETHUSDT_long_liquidation_usd',
    'bybit_ETHUSDT_short_liquidation_usd',
    'bybit_ETHUSDT_liq_delta',

    'bybit_sol_oi_change_pct',
    'bybit_sol_vol_change_pct',
    'bybit_SOLUSDT_ls_ratio',
    'bybit_SOLUSDT_taker_delta',
    'bybit_SOLUSDT_long_liquidation_usd',
    'bybit_SOLUSDT_short_liquidation_usd',
    'bybit_SOLUSDT_liq_delta',

    # ======================================

    'Binance_btc_oi_change_pct',
    'Binance_btc_vol_change_pct',
    'Binance_BTCUSDT_ls_ratio',
    'Binance_BTCUSDT_taker_delta',
    'Binance_BTCUSDT_long_liquidation_usd',
    'Binance_BTCUSDT_short_liquidation_usd',
    'Binance_BTCUSDT_liq_delta',

    'Binance_eth_oi_change_pct',
    'Binance_eth_vol_change_pct',
    'Binance_ETHUSDT_ls_ratio',
    'Binance_ETHUSDT_taker_delta',
    'Binance_ETHUSDT_long_liquidation_usd',
    'Binance_ETHUSDT_short_liquidation_usd',
    'Binance_ETHUSDT_liq_delta',

    'Binance_sol_oi_change_pct',
    'Binance_sol_vol_change_pct',
    'Binance_SOLUSDT_ls_ratio',
    'Binance_SOLUSDT_taker_delta',
    'Binance_SOLUSDT_long_liquidation_usd',
    'Binance_SOLUSDT_short_liquidation_usd',
    'Binance_SOLUSDT_liq_delta',

]
def predict_next_day(csv_path):
    # 2. Подготовка свежих данных
    df_fresh = pd.read_csv(csv_path)
    df_fresh = df_fresh.sort_values('time').tail(35)  # Берем чуть больше window_size

    # Повторяем Feature Engineering (точно так же, как в обучении!)
    df_fresh['target_pct'] = df_fresh['Binance_BTCUSDT_price'].pct_change() * 100
    
    
    df_fresh['bybit_btc_oi_change_pct'] = df_fresh['bybit_BTCUSDT_oi'].pct_change() * 100
    df_fresh['bybit_btc_vol_change_pct'] = df_fresh['bybit_BTCUSDT_volume_usd'].pct_change() * 100
    df_fresh['bybit_BTCUSDT_ls_ratio'] = df_fresh['bybit_BTCUSDT_ls_ratio'].pct_change() * 100
    df_fresh['bybit_BTCUSDT_taker_delta'] = df_fresh['bybit_BTCUSDT_taker_delta'].pct_change() * 100
    df_fresh['bybit_BTCUSDT_long_liquidation_usd'] = df_fresh['bybit_BTCUSDT_long_liquidation_usd'].pct_change() * 100
    df_fresh['bybit_BTCUSDT_short_liquidation_usd'] = df_fresh['bybit_BTCUSDT_short_liquidation_usd'].pct_change() * 100
    df_fresh['bybit_BTCUSDT_liq_delta'] = df_fresh['bybit_BTCUSDT_liq_delta'].pct_change() * 100

    df_fresh['bybit_eth_oi_change_pct'] = df_fresh['bybit_ETHUSDT_oi'].pct_change() * 100
    df_fresh['bybit_eth_vol_change_pct'] = df_fresh['bybit_ETHUSDT_volume_usd'].pct_change() * 100
    df_fresh['bybit_ETHUSDT_ls_ratio'] = df_fresh['bybit_ETHUSDT_ls_ratio'].pct_change() * 100
    df_fresh['bybit_ETHUSDT_taker_delta'] = df_fresh['bybit_ETHUSDT_taker_delta'].pct_change() * 100
    df_fresh['bybit_ETHUSDT_long_liquidation_usd'] = df_fresh['bybit_ETHUSDT_long_liquidation_usd'].pct_change() * 100
    df_fresh['bybit_ETHUSDT_short_liquidation_usd'] = df_fresh['bybit_ETHUSDT_short_liquidation_usd'].pct_change() * 100
    df_fresh['bybit_ETHUSDT_liq_delta'] = df_fresh['bybit_ETHUSDT_liq_delta'].pct_change() * 100
    
    

    df_fresh['bybit_sol_oi_change_pct'] = df_fresh['bybit_SOLUSDT_oi'].pct_change() * 100
    df_fresh['bybit_sol_vol_change_pct'] = df_fresh['bybit_SOLUSDT_volume_usd'].pct_change() * 100
    df_fresh['bybit_SOLUSDT_ls_ratio'] = df_fresh['bybit_SOLUSDT_ls_ratio'].pct_change() * 100
    df_fresh['bybit_SOLUSDT_taker_delta'] = df_fresh['bybit_SOLUSDT_taker_delta'].pct_change() * 100
    df_fresh['bybit_SOLUSDT_long_liquidation_usd'] = df_fresh['bybit_SOLUSDT_long_liquidation_usd'].pct_change() * 100
    df_fresh['bybit_SOLUSDT_short_liquidation_usd'] = df_fresh['bybit_SOLUSDT_short_liquidation_usd'].pct_change() * 100
    df_fresh['bybit_SOLUSDT_liq_delta'] = df_fresh['bybit_SOLUSDT_liq_delta'].pct_change() * 100
    # ======================================
    # ... добавьте здесь расчеты для ETH и SOL из вашего списка features_list ...

    df_fresh['Binance_btc_oi_change_pct'] = df_fresh['Binance_BTCUSDT_oi'].pct_change() * 100
    df_fresh['Binance_btc_vol_change_pct'] = df_fresh['Binance_BTCUSDT_volume_usd'].pct_change() * 100
    df_fresh['Binance_BTCUSDT_ls_ratio'] = df_fresh['Binance_BTCUSDT_ls_ratio'].pct_change() * 100
    df_fresh['Binance_BTCUSDT_taker_delta'] = df_fresh['Binance_BTCUSDT_taker_delta'].pct_change() * 100
    df_fresh['Binance_BTCUSDT_long_liquidation_usd'] = df_fresh['Binance_BTCUSDT_long_liquidation_usd'].pct_change() * 100
    df_fresh['Binance_BTCUSDT_short_liquidation_usd'] = df_fresh['Binance_BTCUSDT_short_liquidation_usd'].pct_change() * 100
    df_fresh['Binance_BTCUSDT_liq_delta'] = df_fresh['Binance_BTCUSDT_liq_delta'].pct_change() * 100

    df_fresh['Binance_eth_oi_change_pct'] = df_fresh['Binance_ETHUSDT_oi'].pct_change() * 100
    df_fresh['Binance_eth_vol_change_pct'] = df_fresh['Binance_ETHUSDT_volume_usd'].pct_change() * 100
    df_fresh['Binance_ETHUSDT_ls_ratio'] = df_fresh['Binance_ETHUSDT_ls_ratio'].pct_change() * 100
    df_fresh['Binance_ETHUSDT_taker_delta'] = df_fresh['Binance_ETHUSDT_taker_delta'].pct_change() * 100
    df_fresh['Binance_ETHUSDT_long_liquidation_usd'] = df_fresh['Binance_ETHUSDT_long_liquidation_usd'].pct_change() * 100
    df_fresh['Binance_ETHUSDT_short_liquidation_usd'] = df_fresh['Binance_ETHUSDT_short_liquidation_usd'].pct_change() * 100
    df_fresh['Binance_ETHUSDT_liq_delta'] = df_fresh['Binance_ETHUSDT_liq_delta'].pct_change() * 100

    df_fresh['Binance_sol_oi_change_pct'] = df_fresh['Binance_SOLUSDT_oi'].pct_change() * 100
    df_fresh['Binance_sol_vol_change_pct'] = df_fresh['Binance_SOLUSDT_volume_usd'].pct_change() * 100
    df_fresh['Binance_SOLUSDT_ls_ratio'] = df_fresh['Binance_SOLUSDT_ls_ratio'].pct_change() * 100
    df_fresh['Binance_SOLUSDT_taker_delta'] = df_fresh['Binance_SOLUSDT_taker_delta'].pct_change() * 100
    df_fresh['Binance_SOLUSDT_long_liquidation_usd'] = df_fresh['Binance_SOLUSDT_long_liquidation_usd'].pct_change() * 100
    df_fresh['Binance_SOLUSDT_short_liquidation_usd'] = df_fresh['Binance_SOLUSDT_short_liquidation_usd'].pct_change() * 100
    df_fresh['Binance_SOLUSDT_liq_delta'] = df_fresh['Binance_SOLUSDT_liq_delta'].pct_change() * 100

    # Очищаем NaN после pct_change
    df_fresh.dropna(inplace=True)

    # 3. Выделяем признаки (последние 30 дней)
    # Список колонок должен СТРОГО совпадать с features_list при обучении
    features_input = df_fresh[features_list].tail(30).values

    # 4. Нормализация
    scaled_input = scaler_x.transform(features_input)  # Используем transform, а не fit_transform!

    # Конвертация в тензор (Batch_size=1, Seq_len=30, Features=22)
    input_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0).to(device)

    # 5. Предсказание
    with torch.no_grad():
        prediction_scaled = model(input_tensor)

    # Обратное масштабирование
    prediction_pct = scaler_y.inverse_transform(prediction_scaled.cpu().numpy())

    return prediction_pct[0][0]


# Запуск
result = predict_next_day("data/data_multiexchange__ethbtcsol_11.01.26.csv")
print(f"--- ПРОГНОЗ НА СЛЕДУЮЩИЙ ДЕНЬ ---")
print(f"Ожидаемое изменение цены BTC: {result:.2f}%")
if result > 0:
    print("Рекомендация: LONG (Вверх)")
else:
    print("Рекомендация: SHORT (Вниз)")