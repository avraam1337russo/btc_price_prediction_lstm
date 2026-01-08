# import requests
# import pandas as pd
# import matplotlib.pyplot as plt
# from datetime import datetime, timedelta
#
# # --- НАСТРОЙКИ ---
# API_KEY = os.getenv('KEY')
# SYMBOL = "BTCUSDT"
# EXCHANGE = "Binance"
# INTERVAL = "1d"
# DAYS_BACK = 365  # Начнем с года, так как история стакана часто ограничена на бесплатных тарифах
#
# HEADERS = {"accept": "application/json", "CG-API-KEY": API_KEY}
#
#
# def fetch_coinglass(endpoint, params):
#     url = f"https://open-api-v4.coinglass.com{endpoint}"
#     try:
#         response = requests.get(url, params=params, headers=HEADERS)
#         res = response.json()
#         if res.get("code") == "0" and res.get("data"):
#             data = res["data"]
#             return pd.DataFrame(data if isinstance(data, list) else data.get("list", []))
#     except Exception as e:
#         print(f"Ошибка {endpoint}: {e}")
#     return pd.DataFrame()
#
#
# # 1. Загрузка данных
# common_params = {"symbol": SYMBOL, "interval": INTERVAL, "exchange": EXCHANGE, "limit": 1000}
#
# print("Загрузка Price...")
# df_p = fetch_coinglass("/api/futures/price/history", common_params)
# print("Загрузка Open Interest...")
# df_oi = fetch_coinglass("/api/futures/open-interest/history", common_params)
# print("Загрузка Orderbook Delta...")
# # Используем историю Ask/Bid в диапазоне 1% (range=1)
# ob_params = {**common_params, "range": "1"}
# df_ob = fetch_coinglass("/api/spot/orderbook/ask-bids-history", ob_params)
#
# # 2. Обработка
# if not df_p.empty and not df_oi.empty and not df_ob.empty:
#     for df in [df_p, df_oi, df_ob]:
#         df['time'] = pd.to_datetime(df['time'], unit='ms')
#
#     # Рассчитываем Orderbook Delta (Bids - Asks)
#     # В этом эндпоинте часто приходят поля 'bidVol' и 'askVol'
#     #
#     #TODO
#     df_ob['ob_delta'] = pd.to_numeric(df_ob['bidVol']) - pd.to_numeric(df_ob['askVol'])
#
#     # Объединение
#     final = df_p[['time', 'close']].rename(columns={'close': 'price'})
#     final = final.merge(df_oi[['time', 'close']], on='time').rename(columns={'close': 'oi'})
#     final = final.merge(df_ob[['time', 'ob_delta']], on='time')
#     final[['price', 'oi', 'ob_delta']] = final[['price', 'oi', 'ob_delta']].apply(pd.to_numeric)
#
#     # 3. Визуализация (3 подобласти)
#     fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
#
#     # Цена
#     ax1.plot(final['time'], final['price'], color='royalblue', label='BTC Price')
#     ax1.set_ylabel('Price (USD)')
#     ax1.grid(True, alpha=0.3)
#     ax1.legend()
#
#     # Open Interest
#     ax2.plot(final['time'], final['oi'], color='darkorange', label='Open Interest')
#     ax2.set_ylabel('OI (USD)')
#     ax2.grid(True, alpha=0.3)
#     ax2.legend()
#
#     # Orderbook Delta (Bids vs Asks)
#     colors = ['green' if x > 0 else 'red' for x in final['ob_delta']]
#     ax3.bar(final['time'], final['ob_delta'], color=colors, alpha=0.6, label='OB Delta (Bids - Asks)')
#     ax3.set_ylabel('OB Delta Vol')
#     ax3.axhline(0, color='black', lw=1)
#     ax3.grid(True, alpha=0.3)
#     ax3.legend()
#
#     plt.suptitle(f"BTC Analysis: Price, OI and Orderbook Delta ({EXCHANGE})", fontsize=16)
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()
# else:
#     print("Не удалось собрать данные. Проверьте лимиты API и доступность Orderbook History для вашего ключа.")

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


# 1. Загрузка данных (Цена и Открытый интерес)
common_params = {"symbol": SYMBOL, "interval": INTERVAL, "exchange": EXCHANGE, "limit": 1000}

print("Загрузка Price...")
df_p = fetch_coinglass("/api/futures/price/history", common_params)

print("Загрузка Open Interest...")
df_oi = fetch_coinglass("/api/futures/open-interest/history", common_params)



# https://open-api-v4.coinglass.com/api/futures/global-long-short-account-ratio/history
print("Загрузка Long/Short Ratio...")
# Эндпоинт для истории соотношения лонгов и шортов
df_ls = fetch_coinglass("/api/futures/global-long-short-account-ratio/history", common_params)



print("Загрузка Taker Buy/Sell Volume...")
df_taker = fetch_coinglass("/api/futures/v2/taker-buy-sell-volume/history", common_params)


print("Загрузка LONG/SHORT liquidations...")
df_liquidation = fetch_coinglass("/api/futures/liquidation/history", common_params)

if not df_liquidation.empty:
    # 1. Приводим время
    df_liquidation['time'] = pd.to_datetime(df_liquidation['time'], unit='ms')

    # 2. Преобразуем значения в числа (из строк в float)
    df_liquidation['long_liquidation_usd'] = pd.to_numeric(df_liquidation['long_liquidation_usd'])
    df_liquidation['short_liquidation_usd'] = pd.to_numeric(df_liquidation['short_liquidation_usd'])

    # 3. (Опционально) Считаем общие ликвидации и дельту ликвидаций
    df_liquidation['total_liquidations'] = df_liquidation['long_liquidation_usd'] + df_liquidation[
        'short_liquidation_usd']
    df_liquidation['liq_delta'] = df_liquidation['long_liquidation_usd'] - df_liquidation['short_liquidation_usd']
# 2. Обработка и объединение
if not df_p.empty and not df_oi.empty and not df_ls.empty:
    # Приводим время к формату datetime
    df_p['time'] = pd.to_datetime(df_p['time'], unit='ms')
    df_oi['time'] = pd.to_datetime(df_oi['time'], unit='ms')
    df_ls['time'] = pd.to_datetime(df_ls['time'], unit='ms')
    df_taker['time'] = pd.to_datetime(df_taker['time'], unit='ms')

    # Преобразуем значения в числа
    df_taker['taker_buy_volume_usd'] = pd.to_numeric(df_taker['taker_buy_volume_usd'])
    df_taker['taker_sell_volume_usd'] = pd.to_numeric(df_taker['taker_sell_volume_usd'])

    # Считаем дельту (чистый приток/отток рыночных ордеров)
    df_taker['taker_delta'] = df_taker['taker_buy_volume_usd'] - df_taker['taker_sell_volume_usd']


    # Очистка и переименование колонок
    df_p = df_p[['time', 'open', 'high', 'low', 'close', 'volume_usd']].rename(columns={'close': 'price'})
    df_oi = df_oi[['time', 'close']].rename(columns={'close': 'oi'})

    # Из L/S Ratio берем колонку 'ratio' (или 'longShortRatio' в зависимости от версии API)
    # Обычно в Coinglass V4 это колонка 'longShortRatio' или 'v'
    # Проверим наличие колонки и переименуем
    ls_col = 'global_account_long_short_ratio' if 'global_account_long_short_ratio' in df_ls.columns else df_ls.columns[1]
    df_ls = df_ls[['time', ls_col]].rename(columns={ls_col: 'ls_ratio'})

    # Последовательное объединение
    final = pd.merge(df_p, df_oi, on='time', how='inner')
    final = pd.merge(final, df_ls, on='time', how='inner')
    final = pd.merge(final, df_taker[['time', 'taker_buy_volume_usd', 'taker_sell_volume_usd', 'taker_delta']],
                     on='time', how='inner')

    # Добавляем ликвидации в итоговый датафрейм
    final = pd.merge(final, df_liquidation[
        ['time', 'long_liquidation_usd', 'short_liquidation_usd', 'total_liquidations', 'liq_delta']],
                     on='time', how='inner')

    # Обновим список числовых колонок для модели
    numeric_cols = ['open', 'high', 'low', 'price', 'volume_usd', 'oi', 'ls_ratio',
                    'taker_delta', 'long_liquidation_usd', 'short_liquidation_usd']

    # Убедимся, что все новые колонки числовые
    final[numeric_cols] = final[numeric_cols].apply(pd.to_numeric)

    print(f"Данные собраны. Строк: {len(final)}. Фичей: {len(numeric_cols)}")









    # Преобразуем в числа
    # numeric_cols = ['open', 'high', 'low', 'price', 'volume_usd', 'oi', 'ls_ratio']
    # final[numeric_cols] = final[numeric_cols].apply(pd.to_numeric)

    final = final.sort_values('time').reset_index(drop=True)

    # print(f"Данные собраны. Число фичей: {len(numeric_cols)}. Строк: {len(final)}")
    #
    # print(f"Данные успешно собраны. Строк: {len(final)}")
    # print(final.tail())
    #
    # # 3. Визуализация
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    #
    # # График цены
    # ax1.plot(final['time'], final['price'], color='royalblue', label='BTC Price (Close)')
    # ax1.set_ylabel('Price (USD)')
    # ax1.grid(True, alpha=0.3)
    # ax1.legend()
    #
    # # График Open Interest
    # ax2.plot(final['time'], final['oi'], color='darkorange', label='Open Interest')
    # ax2.set_ylabel('OI (USD)')
    # ax2.grid(True, alpha=0.3)
    # ax2.legend()
    #
    # plt.suptitle(f"BTC Analysis: Price & Open Interest ({EXCHANGE})", fontsize=16)
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.show()

    # Сохранение для модели
    final.to_csv("train_data_bybit_.csv", index=False)
    print("Файл btc_price_oi_ls_taker_data_.csv сохранен.")
else:
    print("Не удалось собрать данные. Проверьте API-ключ или лимиты.")