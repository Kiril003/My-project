import sys, os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
import speech_recognition as sr
import threading, subprocess
import queue
import speech_recognition as sr
from vosk import Model, KaldiRecognizer
import pyaudio
import json
import requests
import time
from ukrainian_tts.tts import TTS, Voices, Stress
import tempfile
import pygame
import wikipedia
import geocoder, librosa
from scipy.signal import butter, lfilter
import numpy as np
import webrtcvad
import noisereduce as nr
import webbrowser
from concurrent.futures import ThreadPoolExecutor

def class_method(func):
    def wrapper(*args, **kwargs):
        if len(args) > 0 and isinstance(args[0], HomeWidget):
            return func(*args, **kwargs)
        else:
            return func(HomeWidget.instance, *args, **kwargs)
    return wrapper

class DependencyLoader(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    message = pyqtSignal(str)

    def run(self):
        # Список залежностей для завантаження
        dependencies = [
            ("Ініціалізація голосового помічника", self.init_voice_assistant),
            ("Завантаження погодних даних", self.init_weather),
            ("Налаштування музичного плеєра", self.init_music_player),
            # Додайте інші залежності за потребою
        ]

        total = len(dependencies)
        for i, (msg, func) in enumerate(dependencies, 1):
            self.message.emit(msg)
            func()
            self.progress.emit(int(i / total * 100))

        self.finished.emit()

    def init_voice_assistant(self):
        # Ініціалізація голосового помічника
        time.sleep(1)  # Імітація завантаження

    def init_weather(self):
        # Ініціалізація погодного сервісу
        time.sleep(1)  # Імітація завантаження

    def init_music_player(self):
        # Ініціалізація музичного плеєра
        time.sleep(1)  # Імітація завантаження

class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Завантаження')
        self.setFixedSize(400, 200)
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.label_animation = QLabel()
        self.movie = QMovie("loading_animation.gif")
        self.label_animation.setMovie(self.movie)
        layout.addWidget(self.label_animation)

        self.label_message = QLabel("Завантаження...")
        self.label_message.setAlignment(Qt.AlignCenter)
        self.label_message.setStyleSheet("font-size: 16pt; color: #ffffff;")
        layout.addWidget(self.label_message)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)

    def start_animation(self):
        self.movie.start()

    def set_progress(self, value):
        self.progress_bar.setValue(value)

    def set_message(self, message):
        self.label_message.setText(message)

class CustomMessageBox(QDialog):
    def __init__(self, parent=None, title="", message="", buttons=["OK"]):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        layout = QVBoxLayout(self)
        
        container = QWidget(self)
        container.setObjectName("container")
        container_layout = QVBoxLayout(container)
        
        title_label = QLabel(title)
        title_label.setObjectName("title")
        container_layout.addWidget(title_label)
        
        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_label.setObjectName("message")
        container_layout.addWidget(message_label)
        
        button_layout = QHBoxLayout()
        for i, button_text in enumerate(buttons):
            button = QPushButton(button_text)
            button.setObjectName("dialogButton")
            button.clicked.connect(lambda _, idx=i: self.button_clicked(idx))
            button_layout.addWidget(button)
        
        container_layout.addLayout(button_layout)
        layout.addWidget(container)
        
        self.setStyleSheet("""
            QDialog {
                background-color: rgba(0, 0, 0, 100);
            }
            #container {
                background-color: #ffffff;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            #title {
                font-size: 18px;
                font-weight: bold;
                color: #333333;
                padding: 10px;
            }
            #message {
                font-size: 14px;
                color: #666666;
                padding: 0 10px 10px 10px;
            }
            #dialogButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                font-weight: bold;
            }
            #dialogButton:hover {
                background-color: #2980b9;
            }
        """)

        self.opacity_effect = QGraphicsOpacityEffect(self)
        container.setGraphicsEffect(self.opacity_effect)
        self.opacity_effect.setOpacity(0)

        self.animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.animation.setDuration(250)
        self.animation.setStartValue(0)
        self.animation.setEndValue(1)

    def button_clicked(self, index):
        self.done(index)

    def showEvent(self, event):
        super().showEvent(event)
        self.animation.start()

    def closeEvent(self, event):
        self.animation.setDirection(QAbstractAnimation.Backward)
        self.animation.finished.connect(self.close_and_accept)
        self.animation.start()
        event.ignore()

    def close_and_accept(self):
        super().closeEvent(QEvent(QEvent.Close))
        self.accept()

class MobileNotification(QWidget):
    def __init__(self, parent=None, title="", message=""):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)

        layout = QVBoxLayout(self)
        
        container = QWidget(self)
        container.setObjectName("container")
        container_layout = QVBoxLayout(container)
        
        title_label = QLabel(title)
        title_label.setObjectName("title")
        container_layout.addWidget(title_label)
        
        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_label.setObjectName("message")
        container_layout.addWidget(message_label)
        
        layout.addWidget(container)
        
        self.setStyleSheet("""
            #container {
                background-color: rgba(255, 255, 255, 0.9);
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            }
            #title {
                font-size: 16px;
                font-weight: bold;
                color: #333333;
                padding: 5px;
            }
            #message {
                font-size: 14px;
                color: #666666;
                padding: 0 5px 5px 5px;
            }
        """)

        self.opacity_effect = QGraphicsOpacityEffect(self)
        container.setGraphicsEffect(self.opacity_effect)
        self.opacity_effect.setOpacity(0)

        self.animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.animation.setDuration(300)
        self.animation.setStartValue(0)
        self.animation.setEndValue(1)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.hide_notification)
        self.timer.setSingleShot(True)

    def show_notification(self):
        screen = QApplication.primaryScreen().availableGeometry()
        self.setGeometry(screen.width() - 320, 40, 300, 100)
        self.show()
        self.animation.start()
        self.timer.start(4000)  # 4 секунди

    def hide_notification(self):
        self.animation.setDirection(QAbstractAnimation.Backward)
        self.animation.finished.connect(self.close)
        self.animation.start()

def show_notification(parent, title, message):
    notification = MobileNotification(parent, title, message)
    notification.show_notification()

class AnimatedIconButton(QPushButton):
    def __init__(self, gif_path, size=QSize(50, 50), parent=None):
        super().__init__(parent)
        self.movie = QMovie(gif_path)
        self.movie.setScaledSize(size)
        self.movie.frameChanged.connect(self.on_frame_changed)
        self.movie.start()

    def on_frame_changed(self):
        self.setIcon(QIcon(self.movie.currentPixmap()))

class CustomSidebarButton(QToolButton):
    def __init__(self, icon_path, tooltip, size=QSize(90, 90), parent=None):
        super().__init__(parent)
        self.setFixedSize(size)
        self.setToolTip(tooltip)
        self.setCheckable(True)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.icon_label = QLabel(self)
        self.icon_label.setFixedSize(size * 0.8)  # Фіксуємо розмір label
        self.set_icon(icon_path, size)
        
        layout.addWidget(self.icon_label, 0, Qt.AlignCenter)
        
        self.setStyleSheet("""
            QToolButton {
                background-color: transparent;
                border: none;
                border-radius: 10px;
            }
            QToolButton:hover {
                background-color: rgba(52, 73, 94, 0.5);
            }
            QToolButton:checked {
                background-color: rgba(52, 152, 219, 0.5);
            }
        """)

    def set_icon(self, icon_path, size):
        if icon_path.lower().endswith('.gif'):
            movie = QMovie(icon_path)
            movie.setScaledSize(size * 1.06)
            self.icon_label.setMovie(movie)
            movie.start()
        else:
            pixmap = QPixmap(icon_path)
            scaled_pixmap = pixmap.scaled(size * 1.06, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.icon_label.setPixmap(scaled_pixmap)
        
        self.icon_label.setAlignment(Qt.AlignCenter)

class VoiceAssistant(QObject):
    speech_recognized = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.recognizer = sr.Recognizer()
        self.is_listening = False
        self.language = "uk-UA"
        self.use_vosk = False
        self.vosk_model = None
        self.audio_queue = queue.Queue()
        self.recognition_thread = None
        self.settings = QSettings("YourCompany", "SmartAssistant")
        self.vosk_models = {
            "uk-UA": self.settings.value("vosk_model_uk", ""),
            "en-US": self.settings.value("vosk_model_en", ""),
            "ru-RU": self.settings.value("vosk_model_ru", "")
        }
        self.last_result = ""  # Зберігаємо останній результат

    def toggle_listening(self, checked):
        if checked:
            self.start_listening()
        else:
            self.stop_listening()

    def start_listening(self):
        if not self.is_listening:
            self.is_listening = True
            self.last_result = ""
            self.audio_queue = queue.Queue()
            self.recognition_thread = threading.Thread(target=self.listen_in_background, daemon=True)
            self.recognition_thread.start()

    def stop_listening(self):
        if self.is_listening:
            self.is_listening = False
            if self.recognition_thread:
                self.recognition_thread.join()
                self.recognition_thread = None

    def listen_in_background(self):
        if not self.use_vosk and self.check_internet_connection():
            self.listen_with_google()
        else:
            self.listen_with_vosk()

    def check_internet_connection(self):
        try:
            requests.get("http://www.google.com", timeout=3)
            return True
        except requests.ConnectionError:
            return False

    def switch_to_vosk(self):
        if not self.use_vosk:
            self.use_vosk = True
            self.error_occurred.emit("Немає з'єднання з інтернетом. Перемикання на офлайн-розпізнавання.")
            self.stop_listening()
            QTimer.singleShot(0, self.start_listening)

    def listen_with_google(self):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            while self.is_listening:
                try:
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                    self.audio_queue.put(audio)
                    threading.Thread(target=self.recognize_google, args=(audio,), daemon=True).start()
                except sr.WaitTimeoutError:
                    pass
                except Exception as e:
                    self.error_occurred.emit(f"Помилка при прослуховуванні: {str(e)}")

    def recognize_google(self, audio):
        try:
            text = self.recognizer.recognize_google(audio, language=self.language)
            if text != self.last_result:  # Перевіряємо, чи результат новий
                self.last_result = text
                self.speech_recognized.emit(text)
        except sr.UnknownValueError:
            pass  # Ігноруємо нерозпізнану мову
        except sr.RequestError:
            self.switch_to_vosk()
        except Exception as e:
            self.error_occurred.emit(f"Помилка при розпізнаванні: {str(e)}")

    def listen_with_vosk(self):
        model_path = self.get_current_vosk_model_path()
        if not model_path or not os.path.exists(model_path):
            self.error_occurred.emit(f"Модель Vosk для {self.language} не знайдена. Будь ласка, встановіть правильний шлях.")
            return

        try:
            self.vosk_model = Model(model_path)
            
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
            stream.start_stream()

            rec = KaldiRecognizer(self.vosk_model, 16000)

            while self.is_listening:
                data = stream.read(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text = result['text']
                    if text and text != self.last_result:
                        self.last_result = text
                        self.speech_recognized.emit(text)

            stream.stop_stream()
            stream.close()
            p.terminate()
        except Exception as e:
            self.error_occurred.emit(f"Помилка при використанні Vosk: {str(e)}")
        finally:
            self.is_listening = False
    def set_vosk_model_path(self, language, path):
        if language in self.vosk_models:
            self.vosk_models[language] = path
            self.settings.setValue(f"vosk_model_{language[:2]}", path)
        else:
            self.error_occurred.emit(f"Непідтримувана мова: {language}")

    def get_current_vosk_model_path(self):
        return self.vosk_models.get(self.language, "")

    def change_language(self, language):
        self.language = language
        if self.use_vosk:
            self.vosk_model = None

    # Add placeholder methods for commands
    def open_music_player(self):
        print("Opening music player")

    def change_theme_to_dark(self):
        print("Changing theme to dark")

    def show_weather(self):
        print("Showing weather")

    def start_timer(self, minutes):
        print(f"Starting timer for {minutes} minutes")

    def search_information(self, query):
        print(f"Searching information about {query}")

class WeatherThread(QThread):
    weather_updated = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.api_key = "78de1db61ffa6efd32239911ca57f068"

    def run(self):
        while True:
            weather_data = self.get_weather()
            if weather_data:
                self.weather_updated.emit(weather_data)
            self.sleep(120)  # Sleep for 2 minutes

    def get_location(self):
        try:
            g = geocoder.ip('me')
            if g.ok:
                return g
            g = geocoder.google('me')
            if g.ok:
                return g
            return None
        except Exception as e:
            print(f"Помилка при отриманні місцезнаходження: {str(e)}")
            return None

    def get_weather(self):
        location = self.get_location()
        if location:
            latitude, longitude = location.latlng
            url = f"http://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={self.api_key}&units=metric&lang=ua"

            try:
                response = requests.get(url)
                if response.status_code == 200:
                    return response.json()
            except Exception as e:
                print(f"Помилка при отриманні погоди: {str(e)}")
        return None

class HomeWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tts_executor = ThreadPoolExecutor(max_workers=1)
        self.voice_assistant = VoiceAssistant(self)
        self.voice_assistant = VoiceAssistant()
        self.weather_thread = WeatherThread(self)
        self.weather_thread.weather_updated.connect(self.update_weather_ui)
        self.init_ui()
        self.weather_thread.start()

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        self.voice_assistant.speech_recognized.connect(self.on_speech_recognized)
        self.voice_assistant.error_occurred.connect(self.show_error_message)

        # Ініціалізація TTS
        self.tts = TTS(device="cpu")
        pygame.mixer.init()

        self.weather_icons = {
            "01d": "sunny.png",
            "01n": "clear_night.png",
            "02d": "partly_cloudy_day.png",
            "02n": "partly_cloudy_night.png",
            "03d": "cloudy.png",
            "03n": "cloudy.png",
            "04d": "overcast.png",
            "04n": "overcast.png",
            "09d": "rain.png",
            "09n": "rain.png",
            "10d": "rain_sun.png",
            "10n": "rain_night.png",
            "11d": "thunderstorm.png",
            "11n": "thunderstorm.png",
            "13d": "snow.png",
            "13n": "snow.png",
            "50d": "mist.png",
            "50n": "mist.png"
        }
        # Лівий блок
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        greeting_label = QLabel("Вітаємо у Розумному Помічнику")
        greeting_label.setStyleSheet("font-size: 24px; color: #ecf0f1; font-weight: bold;")
        left_layout.addWidget(greeting_label, alignment=Qt.AlignCenter)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            padding: 10px;
            border: 1px solid #bdc3c7;
            font-size: 14px;
        """)
        left_layout.addWidget(self.chat_display)

        input_layout = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Введіть ваше повідомлення або команду...")
        self.chat_input.returnPressed.connect(self.send_message)
        self.chat_input.setStyleSheet("""
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 20px;
            padding: 10px;
            font-size: 14px;
        """)
        input_layout.addWidget(self.chat_input)

        send_button = QPushButton(QIcon("C:\\python-apps\\send.png"), "")
        send_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                border-radius: 20px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        send_button.clicked.connect(self.send_message)
        input_layout.addWidget(send_button)

        self.voice_button = QPushButton(QIcon("C:\\python-apps\\pngwing.png"), "")
        self.voice_button.setCheckable(True)
        self.voice_button.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                border-radius: 20px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
            QPushButton:checked {
                background-color: #e74c3c;
            }
        """)
        self.voice_button.toggled.connect(self.toggle_voice)
        input_layout.addWidget(self.voice_button)

        left_layout.addLayout(input_layout)

        # Правий блок
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Блок погоди
        weather_frame = QFrame()
        weather_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(52, 152, 219, 0.7);
                border-radius: 10px;
                padding: 10px;
            }
            QLabel {
                color: white;
            }
        """)
        weather_layout = QVBoxLayout(weather_frame)

        top_weather_layout = QHBoxLayout()

        self.weather_icon = QLabel()
        self.weather_icon.setFixedSize(100, 100)
        top_weather_layout.addWidget(self.weather_icon)

        temp_desc_layout = QVBoxLayout()
        self.temperature_label = QLabel()
        self.temperature_label.setStyleSheet("font-size: 36px; font-weight: bold;")
        temp_desc_layout.addWidget(self.temperature_label)

        self.weather_description = QLabel()
        self.weather_description.setStyleSheet("font-size: 18px;")
        temp_desc_layout.addWidget(self.weather_description)

        top_weather_layout.addLayout(temp_desc_layout)
        weather_layout.addLayout(top_weather_layout)

        self.city_label = QLabel()
        self.city_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        weather_layout.addWidget(self.city_label)

        details_layout = QHBoxLayout()
        self.humidity_label = QLabel()
        self.wind_label = QLabel()
        self.pressure_label = QLabel()
        details_layout.addWidget(self.humidity_label)
        details_layout.addWidget(self.wind_label)
        details_layout.addWidget(self.pressure_label)
        weather_layout.addLayout(details_layout)

        self.feels_like_label = QLabel()
        weather_layout.addWidget(self.feels_like_label)

        right_layout.addWidget(weather_frame)

        # Блок Вікіпедії
        wiki_frame = QFrame()
        wiki_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(46, 204, 113, 0.7);
                border-radius: 10px;
                padding: 10px;
            }
        """)
        wiki_layout = QVBoxLayout(wiki_frame)

        wiki_input = QLineEdit()
        wiki_input.setPlaceholderText("Введіть запит для пошуку в Вікіпедії...")
        wiki_input.setStyleSheet("""
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 20px;
            padding: 10px;
            font-size: 14px;
        """)
        wiki_layout.addWidget(wiki_input)

        wiki_search_button = QPushButton("Пошук")
        wiki_search_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 20px;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        wiki_search_button.clicked.connect(lambda: self.search_wikipedia(wiki_input.text()))
        wiki_layout.addWidget(wiki_search_button)

        self.wiki_result = QTextEdit()
        self.wiki_result.setReadOnly(True)
        self.wiki_result.setStyleSheet("""
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            padding: 10px;
            font-size: 14px;
        """)
        wiki_layout.addWidget(self.wiki_result)

        right_layout.addWidget(wiki_frame)

        # Додаємо віджети до головного лейауту
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        main_layout.addWidget(splitter)

        # Оновлення погоди при запуску
        self.update_weather()

        self.setup_command_processor()
        
    def check_vosk_models(self):
        settings = QSettings("YourCompany", "SmartAssistant")
        default_path = None
        
        for lang in ["uk-UA", "en-US", "ru-RU"]:
            if not self.vosk_models[lang]:
                self.vosk_models[lang] = settings.value(f"vosk_model_{lang[:2]}", default_path)
                settings.setValue(f"vosk_model_{lang[:2]}", self.vosk_models[lang])

    def update_weather(self):
        weather_data = self.weather_thread.get_weather()
        if weather_data:
            self.update_weather_ui(weather_data)

    def update_weather_ui(self, data):
        if data:
            temperature = data['main']['temp']
            feels_like = data['main']['feels_like']
            humidity = data['main']['humidity']
            pressure = data['main']['pressure']
            wind_speed = data['wind']['speed']
            description = data['weather'][0]['description']
            icon_code = data['weather'][0]['icon']
            city = data['name']
            country = data['sys']['country']

            self.city_label.setText(f"{city}, {country}")
            self.temperature_label.setText(f"{temperature:.1f}°C")
            self.weather_description.setText(description.capitalize())
            self.humidity_label.setText(f"Вологість: {humidity}%")
            self.wind_label.setText(f"Вітер: {wind_speed} м/с")
            self.pressure_label.setText(f"Тиск: {pressure} гПа")
            self.feels_like_label.setText(f"Відчувається як: {feels_like:.1f}°C")

            # Оновлення іконки погоди
            self.update_weather_icon(icon_code)

    def update_weather_icon(self, icon_code):
        icon_path = self.weather_icons.get(icon_code, "default.png")
        full_path = os.path.join("weather_icons", icon_path)
        if os.path.exists(full_path):
            pixmap = QPixmap(full_path)
            self.weather_icon.setPixmap(pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            print(f"Іконка не знайдена: {full_path}")

    def set_custom_weather_icon(self, weather_condition, icon_path):
        """
        Встановлює користувацьку іконку для певного погодного стану.
        
        :param weather_condition: Код погодного стану (наприклад, "01d")
        :param icon_path: Шлях до файлу іконки
        """
        if os.path.exists(icon_path):
            self.weather_icons[weather_condition] = icon_path
        else:
            print(f"Файл іконки не знайдено: {icon_path}")

    def get_location(self):
        try:
            # Спроба отримати місцезнаходження за IP
            g = geocoder.ip('me')
            if g.ok:
                return g
            return None
        except Exception as e:
            print(f"Помилка при отриманні місцезнаходження: {str(e)}")
            return None

    def toggle_voice(self, checked):
        if checked:
            self.voice_assistant.start_listening()
            self.chat_display.append("<i>Голосове розпізнавання розпочато...</i>")
        else:
            self.voice_assistant.stop_listening()
            self.chat_display.append("<i>Голосове розпізнавання зупинено.</i>")

    # def process_command(self, command):
    #     if "погода" in command.lower():
    #         self.update_weather()
    #         return "Я оновив інформацію про погоду для вас."
    #     elif "вікіпедія" in command.lower():
    #         query = command.lower().replace("вікіпедія", "").strip()
    #         return self.search_wikipedia(query)
        # Інші команди залишаються без змін
    def search_wikipedia(self, query):
        try:
            wikipedia.set_lang("uk")
            result = wikipedia.summary(query, sentences=3)
            self.wiki_result.setText(result)
            return f"Ось що я знайшов про '{query}' у Вікіпедії. Перевірте праву панель для повної інформації."
        except wikipedia.exceptions.DisambiguationError as e:
            options = e.options[:5]  # Обмежуємо до 5 варіантів
            self.wiki_result.setText(f"Уточніть, будь ласка. Можливі варіанти:\n" + "\n".join(options))
            return "Знайдено кілька варіантів. Будь ласка, уточніть свій запит."
        except wikipedia.exceptions.PageError:
            self.wiki_result.setText("На жаль, не вдалося знайти інформацію за вашим запитом.")
            return "Вибачте, я не зміг знайти інформацію за вашим запитом у Вікіпедії."

    def show_error_message(self, message):
        custom_msg = CustomMessageBox(None, "Помилка розпізнавання", message, ["OK"])
        custom_msg.exec_()

    def send_message(self):
        message = self.chat_input.text()
        if message:
            self.chat_display.append(f"<b>Ви:</b> {message}")
            self.chat_input.clear()
            response = self.process_command(message)
            if response:
                self.speaknwrite(response)

    def on_speech_recognized(self, text):
        self.chat_display.append(f"<b>Ви:</b> {text}")
        response = self.process_command(text)
        if response:
            self.speaknwrite(response)

    def process_custom_commands(self, command):
        settings = QSettings("YourCompany", "SmartAssistant")
        custom_commands = settings.value("custom_commands", {})
        for cmd_name, cmd_data in custom_commands.items():
            if cmd_data['trigger'].lower() in command.lower():
                return self.execute_custom_command(cmd_data)

    def execute_custom_command(self, cmd_data):
        for action in cmd_data['actions']:
            if action['type'] == 'open_website':
                webbrowser.open(action['url'])
            elif action['type'] == 'open_app':
                subprocess.Popen(action['path'])
            elif action['type'] == 'speak':
                self.speak(action['text'])
            elif action['type'] == 'key_combo':
                # Тут потрібно реалізувати логіку для натискання комбінації клавіш
                pass
            elif action['type'] == 'delay':
                time.sleep(action['seconds'])
        self.speak(f"Виконано команду: {cmd_data['name']}")

    def generate_and_play_tts(self, text):
        self.tts_executor.submit(self._generate_and_play_tts_worker, text)

    def _generate_and_play_tts_worker(self, text):
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, "temp_audio.wav")
        
        try:
            with open(temp_file_path, 'wb') as output_fp:
                output_text = self.tts.tts(text, Voices.Tetiana.value, Stress.Dictionary.value, output_fp)
            
            pygame.mixer.music.load(temp_file_path)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            time.sleep(0.1)
            
        finally:
            try:
                os.remove(temp_file_path)
            except PermissionError:
                pass
            
            try:
                os.rmdir(temp_dir)
            except OSError:
                pass

    def add_example_commands(self):
        examples = [
            "Привіт",
            "Яка зараз погода?",
            "Відкрий музичний плеєр",
            "Котра година?"
        ]
        self.chat_display.append("<b>Приклади команд:</b>")
        for example in examples:
            self.chat_display.append(f"• {example}")
        self.chat_display.append("")

    def animate_button(self, button):
        if isinstance(button, QPushButton):
            self.button_animation.setTargetObject(button)
            self.button_animation.setStartValue(button.geometry())
            self.button_animation.setEndValue(button.geometry().adjusted(2, 2, -2, -2))
            self.button_animation.start()
            QTimer.singleShot(100, lambda: self.button_animation.setDirection(QPropertyAnimation.Backward))
            QTimer.singleShot(100, self.button_animation.start)

    def setup_command_processor(self):
        write = self._write
        speak = self._speak
        speaknwrite = self._speaknwrite

        def process_command(command):
            query = command.lower()
            if "привіт" in query:
                write("Привіт! Чим я можу допомогти?")
            elif "погода" in query:
                write("Вибачте, я ще не маю доступу до інформації про погоду. Але я працюю над цим!")
            elif "музика" in query:
                speak("Хочете послухати музику? Я можу відкрити для вас музичний плеєр.")
            elif "час" in query:
                speaknwrite(f"Зараз {QTime.currentTime().toString('HH:mm')}")
            else:
                return self.process_custom_commands(command)

        self.process_command = process_command

    def _write(self, text):
        self.chat_display.append(f"<b>Асистент:</b> {text}")

    def _speak(self, text):
        self.generate_and_play_tts(text)

    def _speaknwrite(self, text):
        self._write(text)
        self._speak(text)

class StyleHelper:
    @staticmethod
    def get_base_style():
        return """
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                            stop:0 #3498db, stop:0.5 #2980b9, stop:1 #1abc9c);
            }
            QWidget {
                color: #ecf0f1;
                font-family: 'Roboto', sans-serif;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QLineEdit, QTextEdit {
                background-color: rgba(255, 255, 255, 0.8);
                border: 1px solid #bdc3c7;
                padding: 8px;
                border-radius: 5px;
                color: #333333;
            }
            QLabel {
                color: #ecf0f1;
                font-size: 14px;
            }
            QWidget#white-background QLabel {
                color: #333333;
            }
        """

    @staticmethod
    def get_sidebar_style():
        return """
            QWidget#sidebar {
                background-color: rgba(44, 62, 80, 0.7);
                border-right: 1px solid #34495e;
            }
            QPushButton {
                background-color: transparent;
                border: none;
                border-radius: 25px;
                padding: 10px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: rgba(52, 73, 94, 0.5);
            }
            QPushButton:checked {
                background-color: rgba(52, 152, 219, 0.5);
            }
        """

    @staticmethod
    def get_dark_theme():
        return """
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                            stop:0 #2c3e50, stop:0.5 #34495e, stop:1 #2c3e50);
            }
            QWidget {
                color: #ecf0f1;
            }
            QPushButton {
                background-color: #3498db;
                color: #ffffff;
            }
            QLineEdit, QTextEdit {
                background-color: rgba(52, 73, 94, 0.8);
                color: #ecf0f1;
                border: 1px solid #7f8c8d;
            }
            QLabel {
                color: #ecf0f1;
            }
        """

    @staticmethod
    def get_light_theme():
        return StyleHelper.get_base_style()

class FadeTransition:
    def __init__(self, widget):
        self.widget = widget

    def fade_in(self, index):
        self.widget.setCurrentIndex(index)
        current_widget = self.widget.currentWidget()
        animation = QPropertyAnimation(current_widget, b"windowOpacity")
        animation.setDuration(300)
        animation.setStartValue(0)
        animation.setEndValue(1)
        animation.setEasingCurve(QEasingCurve.InOutQuad)
        animation.start()

class BaseWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(StyleHelper.get_base_style())

class AdvancedMediaPlayer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setup_player()
        self.load_playlist()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Відео віджет
        self.video_widget = QVideoWidget()
        layout.addWidget(self.video_widget)

        # Елементи керування
        control_layout = QHBoxLayout()
        self.play_button = QPushButton(self.style().standardIcon(QStyle.SP_MediaPlay), "")
        self.play_button.clicked.connect(self.play_pause)
        control_layout.addWidget(self.play_button)

        self.stop_button = QPushButton(self.style().standardIcon(QStyle.SP_MediaStop), "")
        self.stop_button.clicked.connect(self.stop)
        control_layout.addWidget(self.stop_button)

        self.next_button = QPushButton(self.style().standardIcon(QStyle.SP_MediaSkipForward), "")
        self.next_button.clicked.connect(self.next_track)
        control_layout.addWidget(self.next_button)

        self.prev_button = QPushButton(self.style().standardIcon(QStyle.SP_MediaSkipBackward), "")
        self.prev_button.clicked.connect(self.prev_track)
        control_layout.addWidget(self.prev_button)

        layout.addLayout(control_layout)

        # Слайдер прогресу та мітки часу
        time_layout = QHBoxLayout()
        self.current_time_label = QLabel("00:00")
        time_layout.addWidget(self.current_time_label)
        
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setRange(0, 0)
        self.progress_slider.sliderMoved.connect(self.set_position)
        time_layout.addWidget(self.progress_slider)
        
        self.total_time_label = QLabel("00:00")
        time_layout.addWidget(self.total_time_label)
        
        layout.addLayout(time_layout)

        # Регулятор гучності
        volume_layout = QHBoxLayout()
        volume_label = QLabel("Гучність:")
        volume_layout.addWidget(volume_label)
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(50)
        self.volume_slider.valueChanged.connect(self.set_volume)
        volume_layout.addWidget(self.volume_slider)
        layout.addLayout(volume_layout)

        # Плейлист
        self.playlist_view = QListWidget()
        self.playlist_view.itemDoubleClicked.connect(self.playlist_item_clicked)
        layout.addWidget(self.playlist_view)

        # Кнопки керування плейлистом
        playlist_buttons_layout = QHBoxLayout()
        add_file_button = QPushButton("Додати файл")
        add_file_button.clicked.connect(self.add_file)
        playlist_buttons_layout.addWidget(add_file_button)

        remove_file_button = QPushButton("Видалити файл")
        remove_file_button.clicked.connect(self.remove_file)
        playlist_buttons_layout.addWidget(remove_file_button)

        clear_playlist_button = QPushButton("Очистити плейлист")
        clear_playlist_button.clicked.connect(self.clear_playlist)
        playlist_buttons_layout.addWidget(clear_playlist_button)

        layout.addLayout(playlist_buttons_layout)

    def setup_player(self):
        self.player = QMediaPlayer()
        self.playlist = QMediaPlaylist()  # Create a new playlist
        self.player.setPlaylist(self.playlist)  # Set the playlist for the player
        self.player.setVideoOutput(self.video_widget)
        self.player.stateChanged.connect(self.media_state_changed)
        self.player.positionChanged.connect(self.position_changed)
        self.player.durationChanged.connect(self.duration_changed)
        self.player.error.connect(self.handle_error)

    def load_playlist(self):
        if os.path.exists("playlist.json"):
            with open("playlist.json", "r") as f:
                playlist_data = json.load(f)
                for item in playlist_data:
                    self.playlist_view.addItem(item["name"])
                    self.playlist.addMedia(QMediaContent(QUrl.fromLocalFile(item["path"])))

    def play_pause(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def stop(self):
        self.player.stop()

    def next_track(self):
        self.playlist.next()

    def prev_track(self):
        self.playlist.previous()

    def playlist_item_clicked(self, item):
        index = self.playlist_view.row(item)
        self.playlist.setCurrentIndex(index)
        self.player.play()

    def add_file(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Media files (*.mp3 *.mp4 *.wav *.avi)")
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        if file_dialog.exec_():
            file_names = file_dialog.selectedFiles()
            for file_name in file_names:
                self.playlist_view.addItem(QFileInfo(file_name).fileName())
                self.playlist.addMedia(QMediaContent(QUrl.fromLocalFile(file_name)))
        self.save_playlist()

    def remove_file(self):
        current_row = self.playlist_view.currentRow()
        if current_row >= 0:
            self.playlist_view.takeItem(current_row)
            self.playlist.removeMedia(current_row)
        self.save_playlist()

    def clear_playlist(self):
        self.playlist_view.clear()
        self.playlist.clear()
        self.save_playlist()

    def set_position(self, position):
        self.player.setPosition(position)

    def set_volume(self, volume):
        self.player.setVolume(volume)

    def media_state_changed(self, state):
        if state == QMediaPlayer.PlayingState:
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def position_changed(self, position):
        self.progress_slider.setValue(position)
        self.update_time_label(self.current_time_label, position)

    def duration_changed(self, duration):
        self.progress_slider.setRange(0, duration)
        self.update_time_label(self.total_time_label, duration)

    def update_time_label(self, label, time):
        minutes = int(time / 60000)
        seconds = int((time % 60000) / 1000)
        label.setText(f"{minutes:02d}:{seconds:02d}")

    def save_playlist(self):
        playlist_data = []
        for i in range(self.playlist_view.count()):
            media = self.player.playlist().media(i)
            playlist_data.append({
                "name": self.playlist_view.item(i).text(),
                "path": media.canonicalUrl().toLocalFile()
            })
        with open("playlist.json", "w") as f:
            json.dump(playlist_data, f)

    def handle_error(self):
        self.play_button.setEnabled(False)
        self.status_bar.showMessage("Error: " + self.player.errorString())

            
class SettingsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.setObjectName("settings-widget")
        
        # Ініціалізуємо кольори за замовчуванням
        self.primary_color = "#3498db"
        self.accent_color = "#2980b9"
        
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(self.create_general_tab(), "Загальні")
        self.tab_widget.addTab(self.create_appearance_tab(), "Зовнішній вигляд")
        self.tab_widget.addTab(self.create_speech_tab(), "Розпізнавання мови")
        self.tab_widget.addTab(self.create_custom_commands_tab(), "Користувацькі команди")
        
        layout.addWidget(self.tab_widget)

        save_button = QPushButton("Зберегти налаштування")
        save_button.clicked.connect(self.save_settings)
        save_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                font-size: 16px;
                margin: 4px 2px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        layout.addWidget(save_button)

        self.load_settings()
        self.apply_styles()

    def create_general_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget(scroll)

        layout = QVBoxLayout(scroll_content)

        group_box = QGroupBox("Основні налаштування")
        group_layout = QFormLayout()

        self.language_combo = QComboBox()
        self.language_combo.addItems(["Українська", "English", "Русский"])
        group_layout.addRow("Мова інтерфейсу:", self.language_combo)

        group_box.setLayout(group_layout)
        layout.addWidget(group_box)

        layout.addStretch()
        scroll.setWidget(scroll_content)
        return scroll

    def create_appearance_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget(scroll)

        layout = QVBoxLayout(scroll_content)

        theme_group = QGroupBox("Тема")
        theme_layout = QFormLayout()

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Світла", "Темна"])
        self.theme_combo.currentTextChanged.connect(self.change_theme)
        theme_layout.addRow("Вибір теми:", self.theme_combo)

        theme_group.setLayout(theme_layout)
        layout.addWidget(theme_group)

        color_group = QGroupBox("Кольорова схема")
        color_layout = QFormLayout()

        self.color_scheme_combo = QComboBox()
        self.color_scheme_combo.addItems(["Синя", "Зелена", "Фіолетова", "Користувацька"])
        self.color_scheme_combo.currentTextChanged.connect(self.change_color_scheme)
        color_layout.addRow("Вибір схеми:", self.color_scheme_combo)

        self.primary_color_button = QPushButton("Вибрати основний колір")
        self.primary_color_button.clicked.connect(self.choose_primary_color)
        self.primary_color_button.setStyleSheet(f"background-color: {self.primary_color};")
        color_layout.addRow("Основний колір:", self.primary_color_button)

        self.accent_color_button = QPushButton("Вибрати акцентний колір")
        self.accent_color_button.clicked.connect(self.choose_accent_color)
        self.accent_color_button.setStyleSheet(f"background-color: {self.accent_color};")
        color_layout.addRow("Акцентний колір:", self.accent_color_button)

        color_group.setLayout(color_layout)
        layout.addWidget(color_group)

        font_group = QGroupBox("Шрифт")
        font_layout = QFormLayout()

        self.font_size_slider = QSlider(Qt.Horizontal)
        self.font_size_slider.setRange(8, 24)
        self.font_size_slider.setValue(12)
        self.font_size_slider.valueChanged.connect(self.change_font_size)
        font_layout.addRow("Розмір шрифту:", self.font_size_slider)

        font_group.setLayout(font_layout)
        layout.addWidget(font_group)

        layout.addStretch()
        scroll.setWidget(scroll_content)
        return scroll

    def create_speech_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget(scroll)

        layout = QVBoxLayout(scroll_content)

        speech_group = QGroupBox("Налаштування розпізнавання мови")
        speech_layout = QFormLayout()

        self.speech_recognition_combo = QComboBox()
        self.speech_recognition_combo.addItems(["Google Speech Recognition", "Vosk (офлайн)"])
        self.speech_recognition_combo.currentTextChanged.connect(self.change_speech_recognition)
        speech_layout.addRow("Метод розпізнавання:", self.speech_recognition_combo)

        self.vosk_model_uk_path = QLineEdit()
        speech_layout.addRow("Українська Vosk модель:", self.vosk_model_uk_path)

        self.vosk_model_en_path = QLineEdit()
        speech_layout.addRow("Англійська Vosk модель:", self.vosk_model_en_path)

        self.vosk_model_ru_path = QLineEdit()
        speech_layout.addRow("Російська Vosk модель:", self.vosk_model_ru_path)

        speech_group.setLayout(speech_layout)
        layout.addWidget(speech_group)

        layout.addStretch()
        scroll.setWidget(scroll_content)
        return scroll

    def create_custom_commands_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget(scroll)

        layout = QVBoxLayout(scroll_content)

        commands_group = QGroupBox("Користувацькі команди")
        commands_layout = QVBoxLayout()

        self.custom_commands_list = QListWidget()
        self.custom_commands_list.itemSelectionChanged.connect(self.update_command_buttons)
        commands_layout.addWidget(self.custom_commands_list)

        buttons_layout = QHBoxLayout()
        self.add_command_button = QPushButton("Додати команду")
        self.add_command_button.clicked.connect(self.add_custom_command)
        buttons_layout.addWidget(self.add_command_button)

        self.edit_command_button = QPushButton("Редагувати")
        self.edit_command_button.clicked.connect(self.edit_custom_command)
        self.edit_command_button.setEnabled(False)
        buttons_layout.addWidget(self.edit_command_button)

        self.delete_command_button = QPushButton("Видалити")
        self.delete_command_button.clicked.connect(self.delete_custom_command)
        self.delete_command_button.setEnabled(False)
        buttons_layout.addWidget(self.delete_command_button)

        commands_layout.addLayout(buttons_layout)
        commands_group.setLayout(commands_layout)
        layout.addWidget(commands_group)

        layout.addStretch()
        scroll.setWidget(scroll_content)
        return scroll

    def update_command_buttons(self):
        selected = bool(self.custom_commands_list.selectedItems())
        self.edit_command_button.setEnabled(selected)
        self.delete_command_button.setEnabled(selected)

    def add_custom_command(self):
        dialog = CustomCommandDialog(self)
        if dialog.exec_():
            command_data = dialog.get_command_data()
            self.custom_commands_list.addItem(command_data['name'])
            self.save_custom_command(command_data)

    def edit_custom_command(self):
        current_item = self.custom_commands_list.currentItem()
        if current_item:
            command_name = current_item.text()
            command_data = self.load_custom_command(command_name)
            dialog = CustomCommandDialog(self, command_data)
            if dialog.exec_():
                new_command_data = dialog.get_command_data()
                current_item.setText(new_command_data['name'])
                self.save_custom_command(new_command_data, old_name=command_name)

    def delete_custom_command(self):
        current_item = self.custom_commands_list.currentItem()
        if current_item:
            command_name = current_item.text()
            reply = QMessageBox.question(self, 'Видалення команди',
                                         f"Ви впевнені, що хочете видалити команду '{command_name}'?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.custom_commands_list.takeItem(self.custom_commands_list.row(current_item))
                self.delete_custom_command_from_settings(command_name)

    def choose_primary_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.primary_color = color.name()
            self.primary_color_button.setStyleSheet(f"background-color: {self.primary_color};")
            self.color_scheme_combo.setCurrentText("Користувацька")
            self.apply_styles()
            self.apply_color_scheme()

    def choose_accent_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.accent_color = color.name()
            self.accent_color_button.setStyleSheet(f"background-color: {self.accent_color};")
            self.color_scheme_combo.setCurrentText("Користувацька")
            self.apply_styles()
            self.apply_color_scheme() 

    def change_theme(self, theme):
        if theme == "Темна":
            self.main_window.setStyleSheet(StyleHelper.get_dark_theme())
        else:
            self.main_window.setStyleSheet(StyleHelper.get_light_theme())

    def apply_color_scheme(self):
        self.main_window.apply_color_scheme(self.primary_color, self.accent_color)

    def change_color_scheme(self, scheme):
        if scheme == "Синя":
            self.primary_color = "#3498db"
            self.accent_color = "#2980b9"
        elif scheme == "Зелена":
            self.primary_color = "#2ecc71"
            self.accent_color = "#27ae60"
        elif scheme == "Фіолетова":
            self.primary_color = "#9b59b6"
            self.accent_color = "#8e44ad"
        elif scheme == "Користувацька":
            # Завантажуємо збережені користувацькі кольори
            settings = QSettings("YourCompany", "SmartAssistant")
            self.primary_color = settings.value("custom_primary_color", self.primary_color)
            self.accent_color = settings.value("custom_accent_color", self.accent_color)
        
        self.primary_color_button.setStyleSheet(f"background-color: {self.primary_color};")
        self.accent_color_button.setStyleSheet(f"background-color: {self.accent_color};")
        self.apply_styles()
        self.apply_color_scheme()  # Додаємо цей виклик

    def change_font_size(self, size):
        font = QFont()
        font.setPointSize(size)
        QApplication.setFont(font)

    def change_language(self, language):
        new_language = {"Українська": "uk-UA", "English": "en-US", "Русский": "ru-RU"}[language]
        self.main_window.home_widget.voice_assistant.change_language(new_language)

    def change_speech_recognition(self, method):
        voice_assistant = self.main_window.home_widget.voice_assistant
        voice_assistant.use_vosk = "Vosk" in method
        voice_assistant.stop_listening()
        voice_assistant.start_listening()

    def load_settings(self):
        settings = QSettings("YourCompany", "SmartAssistant")
        self.theme_combo.setCurrentText(settings.value("theme", "Світла"))
        
        color_scheme = settings.value("color_scheme", "Синя")
        self.color_scheme_combo.setCurrentText(color_scheme)
        
        if color_scheme == "Користувацька":
            self.primary_color = settings.value("custom_primary_color", "#3498db")
            self.accent_color = settings.value("custom_accent_color", "#2980b9")
        else:
            self.primary_color = settings.value("primary_color", "#3498db")
            self.accent_color = settings.value("accent_color", "#2980b9")
        
        self.primary_color_button.setStyleSheet(f"background-color: {self.primary_color};")
        self.accent_color_button.setStyleSheet(f"background-color: {self.accent_color};")
        
        self.font_size_slider.setValue(int(settings.value("font_size", 12)))
        self.language_combo.setCurrentText(settings.value("language", "Українська"))
        self.speech_recognition_combo.setCurrentText(settings.value("speech_recognition", "Google Speech Recognition"))

        voice_assistant = self.main_window.home_widget.voice_assistant
        self.vosk_model_uk_path.setText(settings.value("vosk_model_uk", voice_assistant.vosk_models["uk-UA"]))
        self.vosk_model_en_path.setText(settings.value("vosk_model_en", voice_assistant.vosk_models["en-US"]))
        self.vosk_model_ru_path.setText(settings.value("vosk_model_ru", voice_assistant.vosk_models["ru-RU"]))

        custom_commands = settings.value("custom_commands", {})
        for command_name in custom_commands:
            self.custom_commands_list.addItem(command_name)

    def save_settings(self):
        settings = QSettings("YourCompany", "SmartAssistant")
        settings.setValue("theme", self.theme_combo.currentText())
        settings.setValue("color_scheme", self.color_scheme_combo.currentText())
        
        if self.color_scheme_combo.currentText() == "Користувацька":
            settings.setValue("custom_primary_color", self.primary_color)
            settings.setValue("custom_accent_color", self.accent_color)
        else:
            settings.setValue("primary_color", self.primary_color)
            settings.setValue("accent_color", self.accent_color)
        settings.setValue("font_size", self.font_size_slider.value())
        settings.setValue("language", self.language_combo.currentText())
        settings.setValue("speech_recognition", self.speech_recognition_combo.currentText())

        settings.setValue("vosk_model_uk", self.vosk_model_uk_path.text())
        settings.setValue("vosk_model_en", self.vosk_model_en_path.text())
        settings.setValue("vosk_model_ru", self.vosk_model_ru_path.text())

        self.apply_settings()
        self.apply_styles()
        self.apply_color_scheme()  # Додаємо цей виклик

        show_notification(self, "Налаштування", "Налаштування збережено успішно!")

    def apply_settings(self):
        self.change_theme(self.theme_combo.currentText())
        self.apply_color_scheme()
        self.change_font_size(self.font_size_slider.value())
        self.change_language(self.language_combo.currentText())
        self.change_speech_recognition(self.speech_recognition_combo.currentText())

    def save_custom_command(self, command_data, old_name=None):
        settings = QSettings("YourCompany", "SmartAssistant")
        commands = settings.value("custom_commands", {})
        if old_name and old_name in commands:
            del commands[old_name]
        commands[command_data['name']] = command_data
        settings.setValue("custom_commands", commands)

    def load_custom_command(self, command_name):
        settings = QSettings("YourCompany", "SmartAssistant")
        commands = settings.value("custom_commands", {})
        return commands.get(command_name, {})

    def delete_custom_command_from_settings(self, command_name):
        settings = QSettings("YourCompany", "SmartAssistant")
        commands = settings.value("custom_commands", {})
        if command_name in commands:
            del commands[command_name]
            settings.setValue("custom_commands", commands)

    def apply_styles(self):
        style = f"""
            QWidget#settings-widget {{
                background-color: {self.accent_color};
            }}
            QLabel {{
                color: #f0f0f0;
                font-size: 14px;
            }}
            QComboBox, QLineEdit, QSlider, QSpinBox {{
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 5px;
                padding: 5px;
                font-size: 14px;
                color: #333333;
            }}
            QPushButton {{
                background-color: {self.primary_color};
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: {self.accent_color};
            }}
            QPushButton:disabled {{
                background-color: #bdc3c7;
            }}
            QListWidget {{
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 5px;
                font-size: 14px;
                color: #333333;
            }}
            QGroupBox {{
                font-size: 16px;
                font-weight: bold;
                border: 1px solid #f0f0f0;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                color: #f0f0f0;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
                background-color: {self.primary_color};
            }}
            QTabWidget::pane {{
                border: 1px solid #f0f0f0;
                background-color: {self.accent_color};
            }}
            QTabWidget::tab-bar {{
                left: 5px;
            }}
            QTabBar::tab {{
                background-color: {self.primary_color};
                color: #f0f0f0;
                padding: 8px 12px;
                margin-right: 4px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            QTabBar::tab:selected {{
                background-color: {self.accent_color};
            }}
            QScrollArea {{
                background-color: {self.accent_color};
                border: none;
            }}
        """
        self.setStyleSheet(style)
        
        # Застосовуємо стилі до всіх дочірніх віджетів
        for child in self.findChildren(QWidget):
            child.setStyleSheet(style)

        # Додатково налаштовуємо стилі для конкретних елементів
        for groupbox in self.findChildren(QGroupBox):
            groupbox.setStyleSheet(groupbox.styleSheet() + f"""
                QGroupBox {{
                    color: #f0f0f0;
                    background-color: {self.primary_color};
                    padding: 15px;
                }}
                QGroupBox::title {{
                    subcontrol-origin: margin;
                    left: 7px;
                    padding: 0px 5px 0px 5px;
                    background-color: {self.accent_color};
                }}
            """)

class CustomCommandDialog(QDialog):
    def __init__(self, parent=None, command_data=None):
        super().__init__(parent)
        self.setWindowTitle("Нова команда" if command_data is None else "Редагувати команду")
        self.command_data = command_data or {}
        self.init_ui()
        self.apply_styles()

    def init_ui(self):
        layout = QVBoxLayout(self)

        form_layout = QFormLayout()
        self.name_edit = QLineEdit(self.command_data.get('name', ''))
        self.trigger_edit = QLineEdit(self.command_data.get('trigger', ''))
        form_layout.addRow("Назва команди:", self.name_edit)
        form_layout.addRow("Тригер команди:", self.trigger_edit)
        layout.addLayout(form_layout)

        self.actions_list = QListWidget()
        self.actions_list.setDragDropMode(QListWidget.InternalMove)
        layout.addWidget(QLabel("Послідовність дій:"))
        layout.addWidget(self.actions_list)

        actions_buttons_layout = QHBoxLayout()
        add_action_button = QPushButton("Додати дію")
        add_action_button.clicked.connect(self.add_action)
        actions_buttons_layout.addWidget(add_action_button)

        edit_action_button = QPushButton("Редагувати дію")
        edit_action_button.clicked.connect(self.edit_action)
        actions_buttons_layout.addWidget(edit_action_button)

        delete_action_button = QPushButton("Видалити дію")
        delete_action_button.clicked.connect(self.delete_action)
        actions_buttons_layout.addWidget(delete_action_button)

        layout.addLayout(actions_buttons_layout)

        buttons_layout = QHBoxLayout()
        save_button = QPushButton("Зберегти")
        save_button.clicked.connect(self.accept)
        buttons_layout.addWidget(save_button)

        cancel_button = QPushButton("Скасувати")
        cancel_button.clicked.connect(self.reject)
        buttons_layout.addWidget(cancel_button)

        layout.addLayout(buttons_layout)

        if 'actions' in self.command_data:
            for action in self.command_data['actions']:
                self.actions_list.addItem(self.action_to_string(action))

    def add_action(self):
        dialog = ActionDialog(self)
        if dialog.exec_():
            action_data = dialog.get_action_data()
            self.actions_list.addItem(self.action_to_string(action_data))

    def edit_action(self):
        current_item = self.actions_list.currentItem()
        if current_item:
            index = self.actions_list.row(current_item)
            action_data = self.string_to_action(current_item.text())
            dialog = ActionDialog(self, action_data)
            if dialog.exec_():
                new_action_data = dialog.get_action_data()
                self.actions_list.takeItem(index)
                self.actions_list.insertItem(index, self.action_to_string(new_action_data))

    def delete_action(self):
        current_item = self.actions_list.currentItem()
        if current_item:
            self.actions_list.takeItem(self.actions_list.row(current_item))

    def get_command_data(self):
        actions = []
        for i in range(self.actions_list.count()):
            action_string = self.actions_list.item(i).text()
            actions.append(self.string_to_action(action_string))
        
        return {
            'name': self.name_edit.text(),
            'trigger': self.trigger_edit.text(),
            'actions': actions
        }

    def action_to_string(self, action):
        if action['type'] == 'open_website':
            return f"Відкрити веб-сайт: {action['url']}"
        elif action['type'] == 'open_app':
            return f"Відкрити програму: {action['path']}"
        elif action['type'] == 'speak':
            return f"Озвучити: {action['text']}"
        elif action['type'] == 'key_combo':
            return f"Комбінація клавіш: {'+'.join(action['keys'])}"
        elif action['type'] == 'delay':
            return f"Затримка: {action['seconds']} сек."
        else:
            return "Невідома дія"

    def string_to_action(self, action_string):
        if action_string.startswith("Відкрити веб-сайт:"):
            return {'type': 'open_website', 'url': action_string.split(': ', 1)[1]}
        elif action_string.startswith("Відкрити програму:"):
            return {'type': 'open_app', 'path': action_string.split(': ', 1)[1]}
        elif action_string.startswith("Озвучити:"):
            return {'type': 'speak', 'text': action_string.split(': ', 1)[1]}
        elif action_string.startswith("Комбінація клавіш:"):
            return {'type': 'key_combo', 'keys': action_string.split(': ', 1)[1].split('+')}
        elif action_string.startswith("Затримка:"):
            return {'type': 'delay', 'seconds': float(action_string.split(': ', 1)[1].split()[0])}
        else:
            return {'type': 'unknown'}

    def apply_styles(self):
        self.setStyleSheet("""
            QDialog {
                background-color: #f0f0f0;
            }
            QLabel {
                color: #333333;
                font-size: 14px;
            }
            QLineEdit, QListWidget {
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 5px;
                padding: 5px;
                font-size: 14px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)

class ActionDialog(QDialog):
    def __init__(self, parent=None, action_data=None):
        super().__init__(parent)
        self.setWindowTitle("Нова дія" if action_data is None else "Редагувати дію")
        self.action_data = action_data or {}
        self.init_ui()
        self.apply_styles()

    def init_ui(self):
        layout = QVBoxLayout(self)

        self.action_type_combo = QComboBox()
        self.action_type_combo.addItems(["Відкрити веб-сайт", "Відкрити програму", "Озвучити текст", "Комбінація клавіш", "Затримка"])
        self.action_type_combo.currentTextChanged.connect(self.on_action_type_changed)
        layout.addWidget(QLabel("Тип дії:"))
        layout.addWidget(self.action_type_combo)

        self.stack_layout = QStackedWidget()
        layout.addWidget(self.stack_layout)

        self.url_edit = QLineEdit()
        self.stack_layout.addWidget(self.url_edit)

        self.app_path_edit = QLineEdit()
        self.stack_layout.addWidget(self.app_path_edit)

        self.speak_text_edit = QTextEdit()
        self.stack_layout.addWidget(self.speak_text_edit)

        key_combo_widget = QWidget()
        key_combo_layout = QHBoxLayout(key_combo_widget)
        self.key1_edit = QLineEdit()
        self.key2_edit = QLineEdit()
        self.key3_edit = QLineEdit()
        key_combo_layout.addWidget(self.key1_edit)
        key_combo_layout.addWidget(self.key2_edit)
        key_combo_layout.addWidget(self.key3_edit)
        self.stack_layout.addWidget(key_combo_widget)

        self.delay_spin = QDoubleSpinBox()
        self.delay_spin.setRange(0.1, 60.0)
        self.delay_spin.setSingleStep(0.1)
        self.delay_spin.setValue(1.0)
        self.stack_layout.addWidget(self.delay_spin)

        buttons_layout = QHBoxLayout()
        save_button = QPushButton("Зберегти")
        save_button.clicked.connect(self.accept)
        buttons_layout.addWidget(save_button)

        cancel_button = QPushButton("Скасувати")
        cancel_button.clicked.connect(self.reject)
        buttons_layout.addWidget(cancel_button)

        layout.addLayout(buttons_layout)

        if self.action_data:
            self.set_initial_values()

    def set_initial_values(self):
        if self.action_data['type'] == 'open_website':
            self.action_type_combo.setCurrentText("Відкрити веб-сайт")
            self.url_edit.setText(self.action_data['url'])
        elif self.action_data['type'] == 'open_app':
            self.action_type_combo.setCurrentText("Відкрити програму")
            self.app_path_edit.setText(self.action_data['path'])
        elif self.action_data['type'] == 'speak':
            self.action_type_combo.setCurrentText("Озвучити текст")
            self.speak_text_edit.setText(self.action_data['text'])
        elif self.action_data['type'] == 'key_combo':
            self.action_type_combo.setCurrentText("Комбінація клавіш")
            keys = self.action_data['keys']
            if len(keys) > 0:
                self.key1_edit.setText(keys[0])
            if len(keys) > 1:
                self.key2_edit.setText(keys[1])
            if len(keys) > 2:
                self.key3_edit.setText(keys[2])
        elif self.action_data['type'] == 'delay':
            self.action_type_combo.setCurrentText("Затримка")
            self.delay_spin.setValue(self.action_data['seconds'])

    def on_action_type_changed(self, action_type):
        if action_type == "Відкрити веб-сайт":
            self.stack_layout.setCurrentWidget(self.url_edit)
        elif action_type == "Відкрити програму":
            self.stack_layout.setCurrentWidget(self.app_path_edit)
        elif action_type == "Озвучити текст":
            self.stack_layout.setCurrentWidget(self.speak_text_edit)
        elif action_type == "Комбінація клавіш":
            self.stack_layout.setCurrentWidget(self.stack_layout.widget(3))  # key_combo_widget
        elif action_type == "Затримка":
            self.stack_layout.setCurrentWidget(self.delay_spin)

    def get_action_data(self):
        action_type = self.action_type_combo.currentText()
        if action_type == "Відкрити веб-сайт":
            return {'type': 'open_website', 'url': self.url_edit.text()}
        elif action_type == "Відкрити програму":
            return {'type': 'open_app', 'path': self.app_path_edit.text()}
        elif action_type == "Озвучити текст":
            return {'type': 'speak', 'text': self.speak_text_edit.toPlainText()}
        elif action_type == "Комбінація клавіш":
            keys = [self.key1_edit.text(), self.key2_edit.text(), self.key3_edit.text()]
            return {'type': 'key_combo', 'keys': [k for k in keys if k]}
        elif action_type == "Затримка":
            return {'type': 'delay', 'seconds': self.delay_spin.value()}

    def apply_styles(self):
        self.setStyleSheet("""
            QDialog {
                background-color: #f0f0f0;
            }
            QLabel {
                color: #333333;
                font-size: 14px;
            }
            QComboBox, QLineEdit, QTextEdit, QDoubleSpinBox {
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 5px;
                padding: 5px;
                font-size: 14px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
class MainWindow(QMainWindow):
    def __init__(self, voice_assistant, weather_service, music_player):
        super().__init__()
        self.voice_assistant = voice_assistant
        self.weather_service = weather_service
        self.music_player = music_player
        self.setWindowTitle("Розумний Помічник")
        self.setGeometry(100, 100, 1200, 800)

        self.current_color_scheme = "#3498db" 

        self.setStyleSheet(StyleHelper.get_base_style())

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(80)
        sidebar.setStyleSheet(StyleHelper.get_sidebar_style())
        sidebar_layout = QVBoxLayout(sidebar)

        buttons = [
            ('home', 'Головна', "C:\\python-apps\\home.png"),
            ('music', 'Музичний плеєр', "C:\\python-apps\\music.png"),
            ('settings', 'Налаштування', "C:\\Users\\mylos\\Downloads\\settings (1).gif")
        ]

        self.sidebar_buttons = []
        for icon, tooltip, icon_file in buttons:
            btn = CustomSidebarButton(icon_file, tooltip, QSize(60, 60))
            btn.clicked.connect(lambda checked, b=icon: self.switch_widget(b))
            sidebar_layout.addWidget(btn)
            self.sidebar_buttons.append(btn)


        sidebar_layout.addStretch()

        sidebar_layout.addStretch()
        main_layout.addWidget(sidebar)

        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)

        self.home_widget = HomeWidget()
        self.music_player_widget = AdvancedMediaPlayer()
        self.settings_widget = SettingsWidget(self)

        self.stacked_widget.addWidget(self.home_widget)
        self.stacked_widget.addWidget(self.music_player_widget)
        self.stacked_widget.addWidget(self.settings_widget)

        self.fade_transition = FadeTransition(self.stacked_widget)

        self.switch_widget('home')

        self.load_settings()

    def apply_color_scheme(self, primary_color, accent_color):
        if primary_color != self.current_color_scheme:
            old_color = self.current_color_scheme
            self.current_color_scheme = primary_color
            
            self.setStyleSheet(self.styleSheet().replace(old_color, primary_color))
            
            for widget in self.findChildren(QWidget):
                widget.setStyleSheet(widget.styleSheet().replace(old_color, primary_color))
            
            self.update_specific_widgets(primary_color, accent_color)

    def update_specific_widgets(self, primary_color, accent_color):
        sidebar_style = StyleHelper.get_sidebar_style().replace("#3498db", primary_color)
        self.findChild(QWidget, "sidebar").setStyleSheet(sidebar_style)
        
        for button in self.sidebar_buttons:
            button.setStyleSheet(button.styleSheet().replace("#3498db", primary_color))
    

    def switch_widget(self, widget_name):
        index_map = {'home': 0, 'music': 1, 'settings': 2}
        if widget_name in index_map:
            self.fade_transition.fade_in(index_map[widget_name])
            for i, btn in enumerate(self.sidebar_buttons):
                btn.setChecked(i == index_map[widget_name])

    def load_settings(self):
        settings = QSettings("YourCompany", "SmartAssistant")
        theme = settings.value("theme", "Світла")
        color_scheme = settings.value("color_scheme", "Синя")
        font_size = settings.value("font_size", 12)
        language = settings.value("language", "Українська")

        self.settings_widget.theme_combo.setCurrentText(theme)
        self.settings_widget.color_scheme_combo.setCurrentText(color_scheme)
        self.settings_widget.font_size_slider.setValue(font_size)
        self.settings_widget.language_combo.setCurrentText(language)

        self.settings_widget.change_theme(theme)
        self.settings_widget.change_color_scheme(color_scheme)
        self.settings_widget.change_font_size(font_size)
        self.settings_widget.change_language(language)

    def closeEvent(self, event):
        custom_msg = CustomMessageBox(self, "Вихід", "Ви впевнені, що хочете вийти?", ["Так", "Ні"])
        if custom_msg.exec_() == 0:  # "Так" - перша кнопка (індекс 0)
            event.accept()
        else:
            event.ignore()

class OnboardingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ласкаво просимо!")
        self.setGeometry(200, 200, 400, 300)
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                            stop:0 #3498db, stop:1 #f1c40f);
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
                font-size: 16px;
            }
            QLineEdit, QSpinBox {
                background-color: rgba(255, 255, 255, 0.8);
                border: none;
                padding: 8px;
                border-radius: 5px;
                color: #333333;
            }
            QPushButton {
                background-color: #2c3e50;
                color: #ffffff;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #34495e;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        welcome_label = QLabel("Вітаємо у Розумному Помічнику!")
        welcome_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(welcome_label)
        
        name_label = QLabel("Як вас звати?")
        layout.addWidget(name_label)
        
        self.name_input = QLineEdit()
        layout.addWidget(self.name_input)
        
        age_label = QLabel("Скільки вам років?")
        layout.addWidget(age_label)
        
        self.age_input = QSpinBox()
        self.age_input.setRange(1, 120)
        layout.addWidget(self.age_input)
        
        interests_label = QLabel("Виберіть ваші інтереси:")
        layout.addWidget(interests_label)
        
        interests = ["Музика", "Фільми", "Спорт", "Подорожі", "Технології"]
        self.interest_checkboxes = []
        for interest in interests:
            checkbox = QCheckBox(interest)
            checkbox.setStyleSheet("QCheckBox { color: #ffffff; }")
            layout.addWidget(checkbox)
            self.interest_checkboxes.append(checkbox)
        
        finish_button = QPushButton("Завершити")
        finish_button.clicked.connect(self.accept)
        layout.addWidget(finish_button)

    def get_user_info(self):
        interests = [cb.text() for cb in self.interest_checkboxes if cb.isChecked()]
        user_data = {
            "name": self.name_input.text(),
            "age": self.age_input.value(),
            "interests": interests
        }
        
        return user_data

def main():
    app = QApplication(sys.argv)

    splash = SplashScreen()
    splash.show()
    splash.start_animation()

    # Create instances of required services
    voice_assistant = VoiceAssistant()
    weather_service = WeatherThread()
    music_player = AdvancedMediaPlayer()

    # Створіть QThread для завантаження залежностей
    loader_thread = QThread()
    loader = DependencyLoader()
    loader.moveToThread(loader_thread)
    loader.finished.connect(loader_thread.quit)
    loader.progress.connect(splash.set_progress)
    loader.message.connect(splash.set_message)
    loader_thread.started.connect(loader.run)
    loader_thread.start()

    # Чекаємо, поки завантаження завершиться
    loader_thread.finished.connect(lambda: show_main_window(splash, voice_assistant, weather_service, music_player))

    sys.exit(app.exec_())

def show_main_window(splash, voice_assistant, weather_service, music_player):
    window = MainWindow(voice_assistant, weather_service, music_player)
    window.show()
    splash.close()

if __name__ == '__main__':
    main()