import telebot
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


class SalesBot:
    def __init__(self, token):
        self.bot = telebot.TeleBot(token)
        self.bot.message_handler(commands=['start'])(self.start)
        self.sales_data = []
        self.forecast_days = 0

    def start(self, message):
        self.bot.send_message(
            message.chat.id, "На какое количество дней вам нужен прогноз?")
        self.bot.register_next_step_handler(message, self.get_forecast_days)

    def get_forecast_days(self, message):
        try:
            self.forecast_days = int(message.text)
            self.bot.send_message(
                message.chat.id, "Введите данные о выручке за последние две недели в формате: дд.мм.гг сумма")
            self.bot.register_next_step_handler(message, self.get_sales_data)
        except ValueError:
            self.bot.send_message(
                message.chat.id, "Пожалуйста, введите число.")

    def get_sales_data(self, message):
        lines = message.text.split('\n')
        for line in lines:
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                self.bot.send_message(
                    message.chat.id, "Пожалуйста, введите данные в формате: дд.мм.гг сумма")
                self.bot.register_next_step_handler(
                    message, self.get_sales_data)
                return
            date, sales = parts
            try:
                self.sales_data.append({'date': datetime.strptime(
                    date, '%d.%m.%y'), 'sales': float(sales)})
            except ValueError:
                self.bot.send_message(
                    message.chat.id, "Пожалуйста, введите данные в правильном формате.")
                self.bot.register_next_step_handler(
                    message, self.get_sales_data)
                return
        if len(self.sales_data) < 14:
            self.bot.register_next_step_handler(message, self.get_sales_data)
        else:
            self.process_data(message)

    def process_data(self, message):
        df = pd.DataFrame(self.sales_data)
        df.set_index('date', inplace=True)

        # обучаем модель
        model = ExponentialSmoothing(
            df['sales'], trend='add', seasonal='add', seasonal_periods=7)
        model_fit = model.fit()

        # делаем прогноз на следующую неделю
        forecast = model_fit.forecast(steps=self.forecast_days)

        # оцениваем модель
        mse = mean_squared_error(df['sales'], model_fit.fittedvalues)
        self.bot.send_message(message.chat.id, f'MSE: {mse}')

        # добавляем даты к прогнозу
        last_date = df.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1), periods=self.forecast_days)
        forecast.index = forecast_dates

        # форматируем прогноз для отправки
        forecast_str = "\n".join([f"{date.strftime('%d.%m.%y')}: {
                                 value}" for date, value in forecast.items()])
        self.bot.send_message(
            message.chat.id, "Обратите внимание, что прогнозирование данных не является точной наукой и результаты могут отличаться.")
        self.bot.send_message(message.chat.id, forecast_str)

        # создаем график
        plt.figure(figsize=(10, 5))
        plt.plot(df['sales'], label='Фактические продажи')
        plt.plot(forecast, label='Прогнозируемые продажи')
        plt.legend(loc='best')
        plt.title('Прогноз продаж')
        plt.xlabel('Дата')
        plt.ylabel('Продажи')
        plt.grid(True)

        # сохраняем график в файл
        plt.savefig('forecast.png')

        # отправляем график
        with open('forecast.png', 'rb') as photo:
            self.bot.send_photo(message.chat.id, photo)

    def run(self):
        self.bot.polling()


if __name__ == "__main__":
    BOT_TOKEN = '6872432196:AAHroeyJEZ2qnzZUhpdh_x9nxgrFnVg9QFE'
    bot = SalesBot(BOT_TOKEN)
    bot.run()
