# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# import joblib  # для сохранения параметров масштабирования
#
#
#
#
#
#
#
#
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# import torch
#
# # Загружаем ваш CSV
# df = pd.read_csv("btc_price_oi_ls_taker_data_.csv")
#
# # 1. Создаем целевую переменную: изменение цены в % на СЛЕДУЮЩИЙ день
# # Мы сдвигаем цену назад (shift(-1)), чтобы сегодняшний ряд данных соответствовал завтрашнему результату
# df['target_pct'] = df['price'].pct_change().shift(-1) * 100
#
# # 2. Преобразуем другие показатели в относительные (необязательно, но полезно)
# # Например, изменение Открытого Интереса вместо его абсолютного значения
# df['oi_change_pct'] = df['oi'].pct_change()
# df['volume_change_pct'] = df['volume_usd'].pct_change()
#
# # 3. Удаляем последнюю строку (там NaN в target_pct из-за shift) и первую (NaN из-за pct_change)
# df.dropna(inplace=True)
#
# # 4. Список признаков (Features), которые пойдут в модель
# features = [
#     'target_pct', # само изменение цены тоже признак
#     'oi_change_pct',
#     'ls_ratio',
#     'taker_delta',
#     'long_liquidation_usd',
#     'short_liquidation_usd',
#     'liq_delta'
# ]
#
# # 5. Масштабирование
# # Для % лучше подходит StandardScaler (центрирует вокруг 0)
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(df[features])
#
# # Сохраняем таргет отдельно (он первый в списке features)
# # Нам это понадобится для обучения
#
#
#
#
#
#
#
#
#
#
# # 1. Загрузка собранных данных
# # df = pd.read_csv("../btc_price_oi_data.csv")
# df = pd.read_csv("btc_price_oi_ls_taker_data_.csv")
# df['time'] = pd.to_datetime(df['time'])
# df = df.sort_values('time')
#
# # Выбираем признаки (features) для обучения
# # Мы исключаем 'time', так как LSTM работает с порядком строк, а не с датами
# features = ['open', 'high', 'low', 'price', 'volume_usd', 'oi', 'ls_ratio',
#             'taker_buy_volume_usd', 'taker_sell_volume_usd', 'taker_delta']
# data = df[features].values
#
# # 2. Нормализация (MinMax в диапазон [0, 1])
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_data = scaler.fit_transform(data)
#
# # Сохраним скалер, чтобы потом "разжать" предсказания обратно в доллары
# joblib.dump(scaler, 'scaler.pkl')
#
#
# # 3. Создание окон (Sliding Window)
# def create_windows(data, window_size=30, target_col_idx=3):
#     """
#     window_size: сколько прошлых дней видит модель (например, 30)
#     target_col_idx: индекс колонки 'price' в списке features (у нас это 3)
#     """
#     X = []
#     y = []
#
#     for i in range(window_size, len(data)):
#         # Берем кусок данных от i-window_size до i
#         X.append(data[i - window_size:i, :])
#         # Предсказываем цену на текущий момент i
#         y.append(data[i, target_col_idx])
#
#     return np.array(X), np.array(y)
#
#
# WINDOW_SIZE = 30  # модель учится на истории в 30 дней
# X, y = create_windows(scaled_data, window_size=WINDOW_SIZE)
#
# # 4. Разделение на обучающую и тестовую выборки (80/20)
# # Временные ряды нельзя перемешивать (shuffle=False)!
# split = int(len(X) * 0.8)
#
# X_train, X_test = X[:split], X[split:]
# y_train, y_test = y[:split], y[split:]
#
# print(f"Форма входных данных (X_train): {X_train.shape}")
# # Результат будет (samples, window_size, features)
# print(f"Форма ответов (y_train): {y_train.shape}")

















import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.preprocessing import StandardScaler

# 1. Загрузка и первичная обработка
df = pd.read_csv("data/data_multi_exchange__bybit_ethbtcsol_3_.csv")
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time')

# --- FEATURE ENGINEERING (Создаем относительные величины) ---
# Предсказываем % изменения цены на СЛЕДУЮЩИЙ день
df['target_pct'] = df['Binance_BTCUSDT_price'].pct_change().shift(-1) * 100


# Bybit
# Другие признаки переводим в изменения, чтобы уйти от огромных чисел
df['bybit_btc_oi_change_pct'] = df['bybit_BTCUSDT_oi'].pct_change() * 100
df['bybit_btc_vol_change_pct'] = df['bybit_BTCUSDT_volume_usd'].pct_change() * 100

df['bybit_eth_oi_change_pct'] = df['bybit_ETHUSDT_oi'].pct_change() * 100
df['bybit_eth_vol_change_pct'] = df['bybit_ETHUSDT_volume_usd'].pct_change() * 100

df['bybit_sol_oi_change_pct'] = df['bybit_SOLUSDT_oi'].pct_change() * 100
df['bybit_sol_vol_change_pct'] = df['bybit_SOLUSDT_volume_usd'].pct_change() * 100



# Binance
df['Binance_btc_oi_change_pct'] = df['Binance_BTCUSDT_oi'].pct_change() * 100
df['Binance_btc_vol_change_pct'] = df['Binance_BTCUSDT_volume_usd'].pct_change() * 100

df['Binance_eth_oi_change_pct'] = df['Binance_ETHUSDT_oi'].pct_change() * 100
df['Binance_eth_vol_change_pct'] = df['Binance_ETHUSDT_volume_usd'].pct_change() * 100

df['Binance_sol_oi_change_pct'] = df['Binance_SOLUSDT_oi'].pct_change() * 100
df['Binance_sol_vol_change_pct'] = df['Binance_SOLUSDT_volume_usd'].pct_change() * 100







# Ликвидации и дельту оставляем в USD, но StandardScaler их выровняет
# Удаляем пустые строки, возникшие после pct_change и shift
df.dropna(inplace=True)

# 2. Выбор признаков
# Мы берем те данные, которые модель будет видеть в момент "сегодня"
features_list = [
    'target_pct', # (как история прошлых изменений)
    
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
    
    
    
    
    
    
    
    
    #======================================

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

# Выделяем массив признаков
data_features = df[features_list].values
# Выделяем таргет (target_pct)
data_target = df['target_pct'].values.reshape(-1, 1)

# 3. Нормализация (РАЗДЕЛЬНАЯ)
# Используем StandardScaler, так как он лучше обрабатывает выбросы в ликвидациях
scaler_x = StandardScaler()
scaler_y = StandardScaler()

scaled_x = scaler_x.fit_transform(data_features)
scaled_y = scaler_y.fit_transform(data_target)

# Сохраняем скалеры
joblib.dump(scaler_x, 'scaler_x.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')

# 4. Создание окон (Sliding Window)
def create_windows(x_data, y_data, window_size=30):
    X = []
    y = []
    for i in range(window_size, len(x_data)):
        # Окно признаков за последние 30 дней
        X.append(x_data[i-window_size:i, :])
        # Цель — процентное изменение на i-й день (который следует за окном)
        y.append(y_data[i, 0])
    return np.array(X), np.array(y)

WINDOW_SIZE = 30
X, y = create_windows(scaled_x, scaled_y, window_size=WINDOW_SIZE)

# 5. Разделение на выборки
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Перевод в тензоры PyTorch
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

print(f"Готово! Входной вектор: {X_train_t.shape}") # (Batch, 30, 8)