import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
import requests

BOT_TOKEN = '7479143987:AAFeIVKGxOQlMNUwvXu2D4MrNmPsVN42klM'
ADMIN_ID = 805757425
API_URL = 'https://famous-discrete-lioness.ngrok-free.app'

bot = telebot.TeleBot(BOT_TOKEN)

@bot.message_handler(commands=['start'])
def start(message):
    if message.from_user.id == ADMIN_ID:
        bot.reply_to(message, ("Вітаю, адміністраторе! Використовуйте наступні команди:\n"
                               "/add_permission - Додати дозвіл\n"
                               "/list_permissions - Переглянути всі дозволи\n"
                               "/check_permission - Перевірити статус дозволу\n"
                               "/update_permission - Оновити термін дії дозволу\n"
                               "/delete_permission - Видалити дозвіл"))
    else:
        bot.reply_to(message, "Вітаю! Це бот для Розумного Помічника. Зверніться до адміністратора для отримання доступу.")

def process_device_id(message, action):
    device_id = message.text.strip()
    markup = InlineKeyboardMarkup()
    markup.row_width = 2
    options = {
        "5 хвилини": "5m",
        "Тиждень": "1w",
        "2 тиждні": "2w",
        "Місяць": "1m",
        "3 місяці": "3m",
        "6 місяців": "6m",
        "Рік": "1y"
    }
    for text, duration in options.items():
        markup.add(InlineKeyboardButton(text, callback_data=f"{action}_{device_id}_{duration}"))
    bot.reply_to(message, "Виберіть термін дії:", reply_markup=markup)

@bot.message_handler(commands=['add_permission'])
def add_permission(message):
    if message.from_user.id == ADMIN_ID:
        bot.reply_to(message, "Введіть ID пристрою:")
        bot.register_next_step_handler(message, lambda msg: process_device_id(msg, 'add'))
    else:
        bot.reply_to(message, "У вас немає доступу до цієї команди.")

@bot.message_handler(commands=['update_permission'])
def update_permission(message):
    if message.from_user.id == ADMIN_ID:
        bot.reply_to(message, "Введіть ID пристрою для оновлення:")
        bot.register_next_step_handler(message, lambda msg: process_device_id(msg, 'update'))
    else:
        bot.reply_to(message, "У вас немає доступу до цієї команди.")

@bot.message_handler(commands=['delete_permission'])
def delete_permission(message):
    if message.from_user.id == ADMIN_ID:
        bot.reply_to(message, "Введіть ID пристрою для видалення:")
        bot.register_next_step_handler(message, process_delete_device_id)
    else:
        bot.reply_to(message, "У вас немає доступу до цієї команди.")

def process_delete_device_id(message):
    device_id = message.text.strip()
    try:
        response = requests.delete(f'{API_URL}/delete_permission', json={'device_id': device_id})
        response.raise_for_status()
        data = response.json()
        if data['deleted']:
            bot.reply_to(message, f"Дозвіл для пристрою ID {device_id} видалено.")
        else:
            bot.reply_to(message, "Дозвіл не знайдено або вже був видалений.")
    except requests.RequestException as e:
        bot.reply_to(message, "Помилка при видаленні дозволу. Спробуйте пізніше.")
        print(f"Помилка при видаленні дозволу: {str(e)}")

@bot.callback_query_handler(func=lambda call: call.data.startswith(('add_', 'update_')))
def handle_permission_callback(call):
    action, device_id, duration = call.data.split('_')
    
    try:
        response = requests.post(f'{API_URL}/add_permission', json={'device_id': device_id, 'duration': duration})
        response.raise_for_status()
        data = response.json()
        if action == 'add':
            bot.answer_callback_query(call.id, f"Дозвіл надано до {data['expiration_date'][:16]}")
        elif action == 'update':
            bot.answer_callback_query(call.id, f"Термін дії оновлено до {data['expiration_date'][:16]}")
    except requests.RequestException as e:
        bot.answer_callback_query(call.id, "Помилка. Спробуйте пізніше.")
        print(f"Помилка: {str(e)}")

@bot.message_handler(commands=['list_permissions'])
def list_permissions(message):
    if message.from_user.id == ADMIN_ID:
        try:
            response = requests.get(f'{API_URL}/list_permissions')
            response.raise_for_status()
            data = response.json()
            permissions = "\n".join([f"ID: {perm['device_id']}, Закінчується: {perm['expiration_date']}" for perm in data])
            bot.reply_to(message, f"Список дозволів:\n{permissions}")
        except requests.RequestException as e:
            bot.reply_to(message, "Помилка при отриманні списку дозволів. Спробуйте пізніше.")
            print(f"Помилка при отриманні списку дозволів: {str(e)}")
    else:
        bot.reply_to(message, "У вас немає доступу до цієї команди.")

@bot.message_handler(commands=['check_permission'])
def check_permission(message):
    if message.from_user.id == ADMIN_ID:
        bot.reply_to(message, "Введіть ID пристрою для перевірки статусу:")
        bot.register_next_step_handler(message, process_check_device_id)
    else:
        bot.reply_to(message, "У вас немає доступу до цієї команди.")

def process_check_device_id(message):
    device_id = message.text.strip()
    try:
        response = requests.post(f'{API_URL}/check_permission', json={'device_id': device_id})
        response.raise_for_status()
        data = response.json()
        if data['permission']:
            bot.reply_to(message, f"Дозвіл активний до {data['expiration_date'][:16]}")
        else:
            bot.reply_to(message, "Дозвіл неактивний або не знайдений.")
    except requests.RequestException as e:
        bot.reply_to(message, "Помилка при перевірці дозволу. Спробуйте пізніше.")
        print(f"Помилка при перевірці дозволу: {str(e)}")

if __name__ == '__main__':
    bot.polling()
