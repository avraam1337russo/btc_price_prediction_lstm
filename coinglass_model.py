import joblib
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from windowing_price_oi_coinglass import X_train, y_train, X_test, y_test
import numpy as np

print("Start")
# Конвертируем numpy массивы в тензоры PyTorch
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1) # делаем колонку

X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Создаем загрузчики данных
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=32, shuffle=False)

import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # batch_first=True означает, что вход имеет форму (batch, seq, feature)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)

        # Полносвязный слой для получения итоговой цены
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Инициализация скрытых состояний (h0, c0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Проход через LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Берем выход последнего временного шага: out[:, -1, :]
        out = self.fc(out[:, -1, :])
        return out

#
# # Параметры модели
# model = LSTMModel(input_dim=10, hidden_dim=64, num_layers=2, output_dim=1)
# criterion = nn.MSELoss()  # Функция потерь для регрессии
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
# train_losses = []
# test_losses = []
# # Выбираем устройство: если есть видеокарта NVIDIA, используем CUDA
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# epochs = 50  # Сколько раз модель полностью просмотрит датасет
#
# for epoch in range(epochs):
#     model.train()
#     running_train_loss = 0.0
#     print("Epoch:{}".format(str(epoch)))
#     for batch_X, batch_y in train_loader:
#         batch_X, batch_y = batch_X.to(device), batch_y.to(device)
#
#
#         optimizer.zero_grad()
#
#         outputs = model(batch_X)
#         loss = criterion(outputs.squeeze(), batch_y)
#
#         loss.backward()
#         optimizer.step()
#
#         running_train_loss += loss.item()
#
# model.eval()
#
# with torch.no_grad():
#     # Прогоняем весь тестовый набор через модель
#     predictions_t = model(X_test_t)
#
#     # Переводим тензоры обратно в numpy
#     predictions = predictions_t.numpy()
#     actuals = y_test_t.numpy()
#
# # 2. Обратное масштабирование (Inverse Transform)
# # Наш scaler обучался на 6 колонках, поэтому для обратного преобразования
# # нам нужно создать "пустую" матрицу той же формы.
# scaler = joblib.load('scaler.pkl')
#
#
# def inverse_transform_target(values, scaler, target_idx=3):
#     # Создаем временную матрицу из нулей
#     dummy = np.zeros((len(values), 10))
#     # Подставляем наши предсказания в колонку цены (индекс 3)
#     dummy[:, target_idx] = values.flatten()
#     # Разжимаем
#     inverse = scaler.inverse_transform(dummy)
#     return inverse[:, target_idx]
#
#
# # Разжимаем предсказания и реальные значения
# pred_prices = inverse_transform_target(predictions, scaler)
# actual_prices = inverse_transform_target(actuals, scaler)
#
# # 3. Визуализация
# plt.figure(figsize=(15, 6))
# plt.plot(actual_prices, label='Реальная цена BTC', color='#1f77b4', linewidth=2)
# plt.plot(pred_prices, label='Предсказание модели', color='#ff7f0e', linestyle='--', linewidth=2)
#
# plt.title('Проверка LSTM модели на тестовых данных (Out-of-sample)')
# plt.xlabel('Дни (тестовый период)')
# plt.ylabel('Цена BTC (USD)')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()
#
# # Расчет ошибки в долларах (MAE)
# mae = np.mean(np.abs(pred_prices - actual_prices))
# print(f"Средняя ошибка модели (MAE): ${mae:.2f}")




# --- НАСТРОЙКИ ---
# Теперь input_dim = 8 (features_list из предыдущего шага: target_pct, oi_change, etc.)
INPUT_DIM = 8
HIDDEN_DIM = 64
NUM_LAYERS = 2
OUTPUT_DIM = 1
LEARNING_RATE = 0.0005  # Для процентов лучше чуть ниже, чтобы не "пролетать" минимум
EPOCHS = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Инициализация модели
model = LSTMModel(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=OUTPUT_DIM).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- ЦИКЛ ОБУЧЕНИЯ ---
train_losses = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)

        # Важно: убедитесь, что размерности совпадают (batch_size, 1)
        loss = criterion(outputs, batch_y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {total_loss / len(train_loader):.4f}")


torch.save(model.state_dict(), 'lstm_bybit_.pth')
# Загружаем скалеры (созданные на этапе подготовки данных)
scaler_x = joblib.load('scaler_x.pkl')
scaler_y = joblib.load('scaler_y.pkl')

model.eval()
with torch.no_grad():
    # Предсказания на тестовых данных
    X_test_t = X_test_t.to(device)
    predictions_scaled = model(X_test_t).cpu().numpy()
    actuals_scaled = y_test_t.numpy()

# Обратное масштабирование: из нормализованных чисел обратно в ПРОЦЕНТЫ
pred_pct = scaler_y.inverse_transform(predictions_scaled)
actual_pct = scaler_y.inverse_transform(actuals_scaled)

# Теперь pred_pct — это массив реальных процентов (например, 1.25, -0.5 и т.д.)





plt.figure(figsize=(15, 6))
plt.plot(actual_pct, label='Реальное изменение %', color='royalblue', alpha=0.7)
plt.plot(pred_pct, label='Предсказание модели %', color='orange', linestyle='--', alpha=0.9)

plt.axhline(0, color='black', linewidth=1, alpha=0.5) # Линия нуля
plt.title('Предсказание процентного изменения BTC (Tomorrow Returns)')
plt.xlabel('Дни')
plt.ylabel('Изменение цены (%)')
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()

# Расчет MAE в процентах
mae_pct = np.mean(np.abs(pred_pct - actual_pct))
print(f"Средняя ошибка модели: {mae_pct:.2f}%")

# Расчет точности направления (Directional Accuracy)
same_direction = np.sign(pred_pct) == np.sign(actual_pct)
accuracy = np.mean(same_direction) * 100
print(f"Точность угадывания направления (Up/Down): {accuracy:.2f}%")