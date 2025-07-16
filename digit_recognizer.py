#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Beautiful Digit Recognition System
Improved UI design with better accuracy display
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib
import os

class BeautifulDigitRecognizer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🔍 AI Распознавание Цифр")

        # Центрируем окно
        self.center_window()
        self.root.configure(bg='#f0f0f0')

        # Устанавливаем минимальный размер окна
        self.root.minsize(1000, 700)

        # Привязываем событие изменения размера для перецентровки
        self.root.bind('<Configure>', self.on_window_configure)

        # Настраиваем стиль
        self.setup_styles()

        # Инициализация моделей
        self.cnn_model = None
        self.rf_model = None
        self.svm_model = None

        # Переменные для изображения
        self.current_image = None
        self.processed_images = []

        # Переменные для рисования
        self.draw_window = None
        self.draw_canvas = None
        self.drawing = False
        self.last_x = None
        self.last_y = None

        # Загружаем или создаем модели
        self.load_or_create_models()

        self.setup_ui()

    def setup_styles(self):
        """Настройка стилей интерфейса"""
        style = ttk.Style()

        # Настраиваем тему
        style.theme_use('clam')

        # Основные цвета
        bg_color = '#2c3e50'
        accent_color = '#3498db'
        success_color = '#27ae60'
        warning_color = '#f39c12'
        danger_color = '#e74c3c'

        # Стили для различных элементов
        style.configure('Title.TLabel',
                       font=('Segoe UI', 20, 'bold'),
                       foreground='#2c3e50',
                       background='#f0f0f0')

        style.configure('Header.TLabel',
                       font=('Segoe UI', 12, 'bold'),
                       foreground='#34495e',
                       background='#ecf0f1')

        style.configure('Result.TLabel',
                       font=('Segoe UI', 18, 'bold'),
                       foreground='#27ae60',
                       background='#ecf0f1')

        style.configure('Info.TLabel',
                       font=('Segoe UI', 10),
                       foreground='#7f8c8d',
                       background='#ecf0f1')

        style.configure('Custom.TButton',
                       font=('Segoe UI', 11, 'bold'),
                       padding=(10, 5))

        style.configure('Action.TButton',
                       font=('Segoe UI', 12, 'bold'),
                       padding=(15, 8))

    def load_or_create_models(self):
        """Загружает существующие модели или создает новые"""
        print("🚀 Инициализация системы распознавания...")

        # CNN модель
        cnn_path = "models/enhanced_cnn_model.h5"
        if os.path.exists(cnn_path):
            try:
                self.cnn_model = keras.models.load_model(cnn_path)
                print("✅ CNN модель загружена")
            except:
                self.cnn_model = self.create_enhanced_cnn()
        else:
            self.cnn_model = self.create_enhanced_cnn()

        # Random Forest модель
        rf_path = "models/enhanced_rf_model.pkl"
        if os.path.exists(rf_path):
            try:
                self.rf_model = joblib.load(rf_path)
                print("✅ Random Forest модель загружена")
            except:
                self.rf_model = self.create_rf_model()
        else:
            self.rf_model = self.create_rf_model()

        # SVM модель
        svm_path = "models/enhanced_svm_model.pkl"
        if os.path.exists(svm_path):
            try:
                self.svm_model = joblib.load(svm_path)
                print("✅ SVM модель загружена")
            except:
                self.svm_model = self.create_svm_model()
        else:
            self.svm_model = self.create_svm_model()

    def create_enhanced_cnn(self):
        """Создает улучшенную CNN модель"""
        print("🧠 Создание улучшенной CNN модели...")

        # Загружаем MNIST данные
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        # Расширяем данные с помощью аугментации
        x_train_augmented, y_train_augmented = self.augment_data(x_train, y_train)

        # Нормализация
        x_train_augmented = x_train_augmented.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        # Reshape для CNN
        x_train_augmented = x_train_augmented.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)

        # Создаем улучшенную CNN модель
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),

            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),

            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.25),

            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(10, activation='softmax')
        ])

        # Компиляция с улучшенными параметрами
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Обучение с callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
        ]

        model.fit(
            x_train_augmented, y_train_augmented,
            epochs=15,
            validation_data=(x_test, y_test),
            callbacks=callbacks,
            verbose=1,
            batch_size=128
        )

        model.save("models/enhanced_cnn_model.h5")
        print("✅ CNN модель сохранена")
        return model

    def augment_data(self, x_train, y_train):
        """Аугментация данных для улучшения модели"""
        print("📈 Аугментация данных...")

        augmented_x = []
        augmented_y = []

        # Оригинальные данные
        augmented_x.extend(x_train)
        augmented_y.extend(y_train)

        # Добавляем повернутые изображения
        for angle in [-15, -10, -5, 5, 10, 15]:
            for i in range(0, len(x_train), 10):  # Каждое 10-е изображение
                rotated = self.rotate_image(x_train[i], angle)
                augmented_x.append(rotated)
                augmented_y.append(y_train[i])

        # Добавляем зашумленные изображения
        for i in range(0, len(x_train), 5):  # Каждое 5-е изображение
            noisy = self.add_noise(x_train[i])
            augmented_x.append(noisy)
            augmented_y.append(y_train[i])

        return np.array(augmented_x), np.array(augmented_y)

    def rotate_image(self, image, angle):
        """Поворот изображения"""
        center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    def add_noise(self, image):
        """Добавление шума"""
        noise = np.random.normal(0, 25, image.shape)
        noisy = image + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def create_rf_model(self):
        """Создает Random Forest модель"""
        print("🌳 Создание Random Forest модели...")

        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        # Извлекаем признаки
        x_train_features = np.array([self.extract_features(img) for img in x_train])

        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(x_train_features, y_train)

        joblib.dump(model, "models/enhanced_rf_model.pkl")
        print("✅ Random Forest модель сохранена")
        return model

    def create_svm_model(self):
        """Создает SVM модель"""
        print("🎯 Создание SVM модели...")

        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        # Используем подвыборку для SVM (быстрее обучение)
        indices = np.random.choice(len(x_train), 10000, replace=False)
        x_train_sample = x_train[indices]
        y_train_sample = y_train[indices]

        # Извлекаем признаки
        x_train_features = np.array([self.extract_features(img) for img in x_train_sample])

        model = SVC(kernel='rbf', probability=True, random_state=42)
        model.fit(x_train_features, y_train_sample)

        joblib.dump(model, "models/enhanced_svm_model.pkl")
        print("✅ SVM модель сохранена")
        return model

    def extract_features(self, image):
        """Извлечение признаков из изображения"""
        features = []

        # Основные статистики
        features.extend([
            np.mean(image), np.std(image), np.min(image), np.max(image)
        ])

        # Плотность пикселей в регионах
        h, w = image.shape
        regions = [
            image[:h//3, :w//3],      # Верх-лево
            image[:h//3, w//3:2*w//3], # Верх-центр
            image[:h//3, 2*w//3:],     # Верх-право
            image[h//3:2*h//3, :w//3], # Центр-лево
            image[h//3:2*h//3, w//3:2*w//3], # Центр
            image[h//3:2*h//3, 2*w//3:], # Центр-право
            image[2*h//3:, :w//3],     # Низ-лево
            image[2*h//3:, w//3:2*w//3], # Низ-центр
            image[2*h//3:, 2*w//3:]    # Низ-право
        ]

        for region in regions:
            features.extend([
                np.mean(region > 128),  # Плотность белых пикселей
                np.sum(region > 128)    # Количество белых пикселей
            ])

        # Проекции
        h_proj = np.sum(image > 128, axis=1)
        v_proj = np.sum(image > 128, axis=0)

        features.extend([
            np.max(h_proj), np.mean(h_proj), np.std(h_proj),
            np.max(v_proj), np.mean(v_proj), np.std(v_proj)
        ])

        return features

    def advanced_preprocessing(self, image):
        """Продвинутая предобработка изображения"""
        if image is None:
            return []

        processed_versions = []

        # Конвертация в градации серого
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Версия 1: Стандартная обработка
        processed_versions.append(self.standard_preprocessing(gray))

        # Версия 2: Улучшенный контраст
        processed_versions.append(self.contrast_preprocessing(gray))

        # Версия 3: Морфологическая обработка
        processed_versions.append(self.morphological_preprocessing(gray))

        # Версия 4: Адаптивная пороговая обработка
        processed_versions.append(self.adaptive_preprocessing(gray))

        return [img for img in processed_versions if img is not None]

    def standard_preprocessing(self, gray):
        """Стандартная предобработка"""
        try:
            # Размытие для удаления шума
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)

            # Пороговая обработка
            if np.mean(blurred) > 127:
                _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            else:
                _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            return self.extract_and_resize(thresh)
        except:
            return None

    def contrast_preprocessing(self, gray):
        """Обработка с улучшением контраста"""
        try:
            # CLAHE для улучшения контраста
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)

            # Пороговая обработка
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            return self.extract_and_resize(thresh)
        except:
            return None

    def morphological_preprocessing(self, gray):
        """Морфологическая обработка"""
        try:
            # Пороговая обработка
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Морфологические операции
            kernel = np.ones((2,2), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

            return self.extract_and_resize(closing)
        except:
            return None

    def adaptive_preprocessing(self, gray):
        """Адаптивная пороговая обработка"""
        try:
            # Адаптивная пороговая обработка
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )

            return self.extract_and_resize(thresh)
        except:
            return None

    def extract_and_resize(self, binary_image):
        """Извлечение цифры и изменение размера"""
        # Находим контуры
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Находим самый большой контур
            largest_contour = max(contours, key=cv2.contourArea)

            # Проверяем размер контура
            if cv2.contourArea(largest_contour) < 50:
                digit_roi = binary_image
            else:
                x, y, w, h = cv2.boundingRect(largest_contour)

                # Добавляем отступы
                margin = max(w, h) // 4
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(binary_image.shape[1] - x, w + 2 * margin)
                h = min(binary_image.shape[0] - y, h + 2 * margin)

                digit_roi = binary_image[y:y+h, x:x+w]
        else:
            digit_roi = binary_image

        # Делаем квадратным и центрируем
        h, w = digit_roi.shape
        size = max(h, w)
        square = np.zeros((size, size), dtype=np.uint8)

        y_offset = (size - h) // 2
        x_offset = (size - w) // 2
        square[y_offset:y_offset+h, x_offset:x_offset+w] = digit_roi

        # Добавляем отступы (20% от размера)
        padded_size = int(size * 1.4)
        padded = np.zeros((padded_size, padded_size), dtype=np.uint8)
        pad_offset = (padded_size - size) // 2
        padded[pad_offset:pad_offset+size, pad_offset:pad_offset+size] = square

        # Изменяем размер до 28x28
        resized = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)

        return resized

    def ensemble_predict(self, processed_images):
        """Ансамблевое предсказание с улучшенной логикой"""
        if not processed_images:
            return None, 0

        all_predictions = []

        for img in processed_images:
            predictions = {}

            # CNN предсказание
            try:
                cnn_input = img.reshape(1, 28, 28, 1).astype('float32') / 255.0
                cnn_pred = self.cnn_model.predict(cnn_input, verbose=0)[0]
                predictions['cnn'] = cnn_pred
            except:
                predictions['cnn'] = np.zeros(10)

            # Random Forest предсказание
            try:
                features = self.extract_features(img).reshape(1, -1)
                rf_pred = self.rf_model.predict_proba(features)[0]
                predictions['rf'] = rf_pred
            except:
                predictions['rf'] = np.zeros(10)

            # SVM предсказание
            try:
                features = self.extract_features(img).reshape(1, -1)
                svm_pred = self.svm_model.predict_proba(features)[0]
                predictions['svm'] = svm_pred
            except:
                predictions['svm'] = np.zeros(10)

            all_predictions.append(predictions)

        # Вычисляем взвешенное среднее с адаптивными весами
        weights = {'cnn': 0.6, 'rf': 0.25, 'svm': 0.15}
        final_prediction = np.zeros(10)

        for pred_dict in all_predictions:
            weighted_pred = np.zeros(10)
            total_weight = 0

            for model_name, weight in weights.items():
                if model_name in pred_dict:
                    weighted_pred += pred_dict[model_name] * weight
                    total_weight += weight

            if total_weight > 0:
                weighted_pred /= total_weight
                final_prediction += weighted_pred

        final_prediction /= len(all_predictions)

        predicted_digit = np.argmax(final_prediction)
        confidence = np.max(final_prediction)

        # Улучшенная логика определения уверенности
        second_best = np.partition(final_prediction, -2)[-2]
        confidence_gap = confidence - second_best

        # Нормализованная уверенность с учетом разрыва
        normalized_confidence = min(100, (confidence * 100) + (confidence_gap * 50))

        return predicted_digit, normalized_confidence, final_prediction

    def setup_ui(self):
        """Настройка красивого интерфейса"""
        # Настраиваем сетку для адаптивности
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Основной контейнер с прокруткой
        main_canvas = tk.Canvas(self.root, bg='#f0f0f0')
        main_canvas.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)

        scrollbar = ttk.Scrollbar(self.root, orient='vertical', command=main_canvas.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        main_canvas.configure(yscrollcommand=scrollbar.set)

        # Основной фрейм с центрированием
        main_frame = tk.Frame(main_canvas, bg='#f0f0f0', padx=15, pady=15)
        self.main_frame = main_frame
        self.main_canvas = main_canvas
        main_canvas.create_window((0, 0), window=main_frame, anchor='n')

        # Заголовок
        header_frame = tk.Frame(main_frame, bg='#f0f0f0')
        header_frame.pack(fill='x', pady=(0, 20))

        title_label = ttk.Label(header_frame, text="🔍 AI Распознавание Цифр", style='Title.TLabel')
        title_label.pack()

        subtitle_label = ttk.Label(header_frame, text="Передовая система машинного обучения",
                                 font=('Segoe UI', 10), foreground='#7f8c8d', background='#f0f0f0')
        subtitle_label.pack()

        # Панель кнопок с центрированием
        buttons_frame = tk.Frame(main_frame, bg='#f0f0f0')
        buttons_frame.pack(fill='x', pady=(0, 15))

        # Внутренний фрейм для центрирования кнопок
        buttons_center = tk.Frame(buttons_frame, bg='#f0f0f0')
        buttons_center.pack(expand=True)

        # Создаем красивые кнопки
        load_btn = tk.Button(buttons_center, text="📂 Загрузить изображение",
                           command=self.load_image,
                           bg='#3498db', fg='white', font=('Segoe UI', 11, 'bold'),
                           padx=20, pady=10, relief='flat', cursor='hand2')
        load_btn.pack(side='left', padx=10)

        recognize_btn = tk.Button(buttons_center, text="🧠 Распознать цифру",
                                command=self.recognize_digit,
                                bg='#27ae60', fg='white', font=('Segoe UI', 11, 'bold'),
                                padx=20, pady=10, relief='flat', cursor='hand2')
        recognize_btn.pack(side='left', padx=10)

        draw_btn = tk.Button(buttons_center, text="✏️ Нарисовать",
                           command=self.open_draw_window,
                           bg='#9b59b6', fg='white', font=('Segoe UI', 11, 'bold'),
                           padx=20, pady=10, relief='flat', cursor='hand2')
        draw_btn.pack(side='left', padx=10)

        clear_btn = tk.Button(buttons_center, text="🗑️ Очистить",
                            command=self.clear_all,
                            bg='#e74c3c', fg='white', font=('Segoe UI', 11, 'bold'),
                            padx=20, pady=10, relief='flat', cursor='hand2')
        clear_btn.pack(side='left', padx=10)

        # Контейнер для изображений
        images_container = tk.Frame(main_frame, bg='#ecf0f1', relief='solid', bd=1)
        images_container.pack(fill='x', pady=(0, 15))

        # Заголовок секции изображений
        images_header = tk.Label(images_container, text="📷 Анализ изображения",
                               font=('Segoe UI', 12, 'bold'), bg='#ecf0f1', fg='#2c3e50')
        images_header.pack(pady=10)

        # Фрейм для изображений с центрированием
        images_frame = tk.Frame(images_container, bg='#ecf0f1')
        images_frame.pack(expand=True, pady=(0, 15))

        # Центральный контейнер для изображений
        images_center = tk.Frame(images_frame, bg='#ecf0f1')
        images_center.pack(expand=True)

        # Оригинальное изображение
        original_container = tk.Frame(images_center, bg='white', relief='solid', bd=2)
        original_container.pack(side='left', padx=15)

        original_title = tk.Label(original_container, text="Оригинальное изображение",
                                font=('Segoe UI', 10, 'bold'), bg='white', fg='#34495e')
        original_title.pack(pady=(8, 5))

        self.original_label = tk.Label(original_container, text="Загрузите изображение\nцифры для анализа",
                                     bg='white', fg='#95a5a6', font=('Segoe UI', 9))
        self.original_label.pack(padx=15, pady=(0, 10))

        # Обработанные изображения
        processed_container = tk.Frame(images_center, bg='white', relief='solid', bd=2)
        processed_container.pack(side='left', padx=15)

        processed_title = tk.Label(processed_container, text="Варианты обработки",
                                 font=('Segoe UI', 10, 'bold'), bg='white', fg='#34495e')
        processed_title.pack(pady=(8, 5))

        processed_grid = tk.Frame(processed_container, bg='white')
        processed_grid.pack(padx=15, pady=(0, 10))

        self.processed_labels = []
        for i in range(4):
            row = i // 2
            col = i % 2
            label = tk.Label(processed_grid, text="—", bg='#f8f9fa', fg='#95a5a6',
                           relief='solid', bd=1)
            label.grid(row=row, column=col, padx=3, pady=3)
            self.processed_labels.append(label)

        # Контейнер результатов
        result_container = tk.Frame(main_frame, bg='#ecf0f1', relief='solid', bd=1)
        result_container.pack(fill='x', pady=(0, 15))

        # Заголовок результатов
        result_header = tk.Label(result_container, text="🎯 Результат распознавания",
                               font=('Segoe UI', 12, 'bold'), bg='#ecf0f1', fg='#2c3e50')
        result_header.pack(pady=10)

        # Основной результат
        self.result_label = tk.Label(result_container, text="Загрузите изображение и нажмите 'Распознать цифру'",
                                   font=('Segoe UI', 16, 'bold'), bg='#ecf0f1', fg='#7f8c8d')
        self.result_label.pack(pady=(0, 15))

        # Детальный анализ
        details_container = tk.Frame(main_frame, bg='#ecf0f1', relief='solid', bd=1)
        details_container.pack(fill='both', expand=True)

        details_header = tk.Label(details_container, text="📊 Детальный анализ",
                                font=('Segoe UI', 12, 'bold'), bg='#ecf0f1', fg='#2c3e50')
        details_header.pack(pady=10)

        # Текстовое поле для деталей
        text_frame = tk.Frame(details_container, bg='#ecf0f1')
        text_frame.pack(fill='both', expand=True, padx=15, pady=(0, 15))

        self.details_text = tk.Text(text_frame, height=6, font=('Consolas', 9),
                                  bg='white', fg='#2c3e50', relief='solid', bd=1)
        details_scrollbar = ttk.Scrollbar(text_frame, orient='vertical', command=self.details_text.yview)
        self.details_text.configure(yscrollcommand=details_scrollbar.set)

        self.details_text.pack(side='left', fill='both', expand=True)
        details_scrollbar.pack(side='right', fill='y')

        # Прогресс бар
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate', style='Custom.Horizontal.TProgressbar')
        self.progress.pack(fill='x', pady=10)

        # Обновляем область прокрутки и центрируем содержимое
        def update_scroll_region(event):
            main_canvas.configure(scrollregion=main_canvas.bbox('all'))
            self.center_content()

        main_frame.bind('<Configure>', update_scroll_region)

        # Привязываем колесико мыши к прокрутке
        def on_mousewheel(event):
            main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        main_canvas.bind("<MouseWheel>", on_mousewheel)

    def load_image(self):
        """Загрузка изображения"""
        file_path = filedialog.askopenfilename(
            title="Выберите изображение цифры",
            filetypes=[
                ("Изображения", "*.png *.jpg *.jpeg *.bmp"),
                ("Все файлы", "*.*")
            ]
        )

        if file_path:
            try:
                self.current_image = cv2.imread(file_path)
                if self.current_image is None:
                    messagebox.showerror("Ошибка", "Не удалось загрузить изображение")
                    return

                self.show_original_image()
                self.preprocess_and_show()

            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при загрузке: {str(e)}")

    def show_original_image(self):
        """Показывает оригинальное изображение"""
        if self.current_image is not None:
            display_image = cv2.resize(self.current_image, (200, 200))
            display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)

            pil_image = Image.fromarray(display_image)
            photo = ImageTk.PhotoImage(pil_image)

            self.original_label.config(image=photo, text="", width=200, height=200)
            self.original_label.image = photo

    def preprocess_and_show(self):
        """Предобработка и показ вариантов"""
        if self.current_image is None:
            return

        self.processed_images = self.advanced_preprocessing(self.current_image)

        # Показываем обработанные варианты
        for i, label in enumerate(self.processed_labels):
            if i < len(self.processed_images) and self.processed_images[i] is not None:
                img = self.processed_images[i]
                display_img = cv2.resize(img, (90, 90), interpolation=cv2.INTER_NEAREST)
                display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB)

                pil_image = Image.fromarray(display_img)
                photo = ImageTk.PhotoImage(pil_image)

                label.config(image=photo, text="", width=90, height=90)
                label.image = photo
            else:
                label.config(image="", text="—", width=90, height=90)
                label.image = None

    def recognize_digit(self):
        """Распознавание цифры"""
        if not self.processed_images:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение")
            return

        try:
            self.progress.start()
            self.root.update()

            # Ансамблевое предсказание
            result = self.ensemble_predict(self.processed_images)

            if result[0] is not None:
                predicted_digit, confidence, probabilities = result

                # Показываем основной результат (убираем отображение уверенности)
                result_text = f"Распознанная цифра: {predicted_digit}"
                self.result_label.config(text=result_text, fg='#27ae60')

                # Показываем детальный анализ
                self.show_detailed_analysis(probabilities, predicted_digit)
            else:
                self.result_label.config(text="Ошибка распознавания", fg='#e74c3c')

            self.progress.stop()

        except Exception as e:
            self.progress.stop()
            messagebox.showerror("Ошибка", f"Ошибка при распознавании: {str(e)}")

    def show_detailed_analysis(self, probabilities, predicted_digit):
        """Показывает детальный анализ"""
        self.details_text.delete(1.0, tk.END)

        analysis = "🎯 РЕЗУЛЬТАТ АНАЛИЗА\n"
        analysis += "=" * 50 + "\n\n"
        analysis += f"✅ Распознанная цифра: {predicted_digit}\n\n"

        analysis += "📊 Вероятности для каждой цифры:\n"
        analysis += "-" * 40 + "\n"

        # Сортируем по убыванию вероятности
        sorted_probs = [(i, prob) for i, prob in enumerate(probabilities)]
        sorted_probs.sort(key=lambda x: x[1], reverse=True)

        for i, (digit, prob) in enumerate(sorted_probs):
            percentage = prob * 100
            if i == 0:  # Первая (самая вероятная)
                bar = "█" * min(20, int(percentage // 2))
                analysis += f"🥇 Цифра {digit}: {percentage:5.1f}% {bar}\n"
            elif i == 1:  # Вторая
                bar = "▓" * min(20, int(percentage // 2))
                analysis += f"🥈 Цифра {digit}: {percentage:5.1f}% {bar}\n"
            elif i == 2:  # Третья
                bar = "▒" * min(20, int(percentage // 2))
                analysis += f"🥉 Цифра {digit}: {percentage:5.1f}% {bar}\n"
            else:
                bar = "░" * min(20, int(percentage // 2))
                analysis += f"   Цифра {digit}: {percentage:5.1f}% {bar}\n"

        analysis += "\n" + "=" * 50 + "\n"
        analysis += f"🔬 Обработано вариантов: {len(self.processed_images)}\n"
        analysis += "🧠 Модели: CNN (60%) + Random Forest (25%) + SVM (15%)\n"
        analysis += "⚡ Метод: Ансамблевое голосование с адаптивными весами\n"
        analysis += "🎨 Предобработка: Множественные алгоритмы обработки\n"

        # Добавляем рекомендации
        top_prob = sorted_probs[0][1] * 100
        second_prob = sorted_probs[1][1] * 100
        gap = top_prob - second_prob

        analysis += "\n" + "💡 АНАЛИЗ КАЧЕСТВА:\n"
        analysis += "-" * 40 + "\n"

        if gap > 40:
            analysis += "✅ Отличная уверенность - результат очень надежный\n"
        elif gap > 20:
            analysis += "👍 Хорошая уверенность - результат надежный\n"
        elif gap > 10:
            analysis += "⚠️  Средняя уверенность - результат вероятно правильный\n"
        else:
            analysis += "❓ Низкая уверенность - возможны альтернативы\n"

        if top_prob > 80:
            analysis += "🎯 Очень высокая вероятность правильности\n"
        elif top_prob > 60:
            analysis += "👌 Высокая вероятность правильности\n"
        elif top_prob > 40:
            analysis += "🤔 Умеренная вероятность правильности\n"
        else:
            analysis += "⚠️  Рекомендуется проверить качество изображения\n"

        self.details_text.insert(tk.END, analysis)

    def clear_all(self):
        """Очистка всех данных"""
        self.current_image = None
        self.processed_images = []

        self.original_label.config(image="", text="Загрузите изображение\nцифры для анализа", width=20, height=10)
        self.original_label.image = None

        for label in self.processed_labels:
            label.config(image="", text="—", width=10, height=5)
            label.image = None

        self.result_label.config(text="Загрузите изображение и нажмите 'Распознать цифру'", fg='#7f8c8d')
        self.details_text.delete(1.0, tk.END)

    def open_draw_window(self):
        """Открывает окно для рисования цифр"""
        if self.draw_window is not None:
            self.draw_window.lift()
            return

        # Создаем новое окно
        self.draw_window = tk.Toplevel(self.root)
        self.draw_window.title("✏️ Нарисуйте цифру")
        self.draw_window.geometry("400x500")
        self.draw_window.configure(bg='#f0f0f0')
        self.draw_window.resizable(False, False)

        # Центрируем окно рисования
        self.center_draw_window()

        # Заголовок
        header = tk.Label(self.draw_window, text="✏️ Нарисуйте цифру от 0 до 9",
                         font=('Segoe UI', 14, 'bold'), bg='#f0f0f0', fg='#2c3e50')
        header.pack(pady=10)

        instruction = tk.Label(self.draw_window, text="Используйте мышь для рисования. Рисуйте крупно и четко!",
                              font=('Segoe UI', 10), bg='#f0f0f0', fg='#7f8c8d')
        instruction.pack(pady=(0, 10))

        # Холст для рисования
        canvas_frame = tk.Frame(self.draw_window, bg='white', relief='solid', bd=2)
        canvas_frame.pack(padx=20, pady=10)

        self.draw_canvas = tk.Canvas(canvas_frame, width=280, height=280, bg='white', cursor='crosshair')
        self.draw_canvas.pack(padx=10, pady=10)

        # Привязываем события мыши
        self.draw_canvas.bind("<Button-1>", self.start_drawing)
        self.draw_canvas.bind("<B1-Motion>", self.draw)
        self.draw_canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        # Кнопки управления
        buttons_frame = tk.Frame(self.draw_window, bg='#f0f0f0')
        buttons_frame.pack(pady=10)

        analyze_btn = tk.Button(buttons_frame, text="🧠 Анализировать",
                               command=self.analyze_drawing,
                               bg='#27ae60', fg='white', font=('Segoe UI', 11, 'bold'),
                               padx=20, pady=8, relief='flat', cursor='hand2')
        analyze_btn.pack(side='left', padx=5)

        clear_draw_btn = tk.Button(buttons_frame, text="🗑️ Очистить холст",
                                  command=self.clear_canvas,
                                  bg='#e74c3c', fg='white', font=('Segoe UI', 11, 'bold'),
                                  padx=20, pady=8, relief='flat', cursor='hand2')
        clear_draw_btn.pack(side='left', padx=5)

        close_btn = tk.Button(buttons_frame, text="❌ Закрыть",
                             command=self.close_draw_window,
                             bg='#95a5a6', fg='white', font=('Segoe UI', 11, 'bold'),
                             padx=20, pady=8, relief='flat', cursor='hand2')
        close_btn.pack(side='left', padx=5)

        # Обработчик закрытия окна
        self.draw_window.protocol("WM_DELETE_WINDOW", self.close_draw_window)

    def center_draw_window(self):
        """Центрирует окно рисования"""
        self.draw_window.update_idletasks()

        # Получаем размеры главного окна
        main_x = self.root.winfo_x()
        main_y = self.root.winfo_y()
        main_width = self.root.winfo_width()
        main_height = self.root.winfo_height()

        # Размеры окна рисования
        draw_width = 400
        draw_height = 500

        # Позиционируем справа от главного окна
        pos_x = main_x + main_width + 20
        pos_y = main_y + (main_height - draw_height) // 2

        # Проверяем, не выходит ли за границы экрана
        screen_width = self.draw_window.winfo_screenwidth()
        if pos_x + draw_width > screen_width:
            pos_x = main_x - draw_width - 20

        self.draw_window.geometry(f"{draw_width}x{draw_height}+{pos_x}+{pos_y}")

    def start_drawing(self, event):
        """Начинает рисование"""
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y

    def draw(self, event):
        """Рисует линию"""
        if self.drawing:
            # Рисуем плавную линию
            self.draw_canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                width=12, fill='black', capstyle=tk.ROUND,
                smooth=tk.TRUE, splinesteps=36
            )
            # Добавляем круги для лучшего соединения линий
            self.draw_canvas.create_oval(
                event.x - 6, event.y - 6, event.x + 6, event.y + 6,
                fill='black', outline='black'
            )
            self.last_x = event.x
            self.last_y = event.y

    def stop_drawing(self, event):
        """Останавливает рисование"""
        self.drawing = False

    def clear_canvas(self):
        """Очищает холст для рисования"""
        self.draw_canvas.delete("all")

    def analyze_drawing(self):
        """Анализирует нарисованную цифру"""
        if self.draw_canvas is None:
            return

        try:
            # Сохраняем изображение с холста
            canvas_image = self.canvas_to_image()

            if canvas_image is not None:
                # Устанавливаем как текущее изображение
                self.current_image = canvas_image

                # Показываем в основном окне
                self.show_original_image()
                self.preprocess_and_show()

                # Закрываем окно рисования
                self.close_draw_window()

                # Поднимаем главное окно
                self.root.lift()
                self.root.focus_force()

                # Показываем сообщение
                self.result_label.config(
                    text="Нарисованная цифра загружена! Нажмите 'Распознать цифру'",
                    fg='#27ae60'
                )
            else:
                self.result_label.config(text="Ошибка: нарисуйте что-нибудь на холсте", fg='#e74c3c')

        except Exception as e:
            self.result_label.config(text=f"Ошибка анализа: {str(e)}", fg='#e74c3c')

    def canvas_to_image(self):
        """Преобразует содержимое холста в изображение"""
        try:
            # Получаем границы рисунка
            bbox = self.draw_canvas.bbox("all")
            if bbox is None:
                return None

            # Создаем белое изображение
            img_array = np.ones((280, 280, 3), dtype=np.uint8) * 255

            # Получаем все элементы холста (линии)
            items = self.draw_canvas.find_all()

            for item in items:
                coords = self.draw_canvas.coords(item)
                if len(coords) >= 4:
                    # Рисуем линии на изображении
                    for i in range(0, len(coords) - 2, 2):
                        x1 = max(0, min(279, int(coords[i])))
                        y1 = max(0, min(279, int(coords[i + 1])))
                        x2 = max(0, min(279, int(coords[i + 2])))
                        y2 = max(0, min(279, int(coords[i + 3])))

                        # Рисуем толстую линию
                        cv2.line(img_array, (x1, y1), (x2, y2), (0, 0, 0), 12)
                        # Добавляем круги на концах для сглаживания
                        cv2.circle(img_array, (x1, y1), 6, (0, 0, 0), -1)
                        cv2.circle(img_array, (x2, y2), 6, (0, 0, 0), -1)

            return img_array

        except Exception as e:
            print(f"Ошибка создания изображения: {e}")
            return None

    def close_draw_window(self):
        """Закрывает окно рисования"""
        if self.draw_window is not None:
            self.draw_window.destroy()
            self.draw_window = None
            self.draw_canvas = None

    def center_window(self):
        """Центрирует окно на экране"""
        # Устанавливаем размер окна
        window_width = 1200
        window_height = 800

        # Обновляем окно для получения актуальных размеров экрана
        self.root.update_idletasks()

        # Получаем размеры экрана
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Вычисляем позицию для центрирования
        pos_x = (screen_width - window_width) // 2
        pos_y = (screen_height - window_height) // 2

        # Устанавливаем геометрию окна
        self.root.geometry(f"{window_width}x{window_height}+{pos_x}+{pos_y}")

    def center_content(self):
        """Центрирует содержимое в канвасе"""
        if hasattr(self, 'main_canvas') and hasattr(self, 'main_frame'):
            self.main_canvas.update_idletasks()

            # Получаем размеры канваса и фрейма
            canvas_width = self.main_canvas.winfo_width()
            frame_width = self.main_frame.winfo_reqwidth()

            # Центрируем содержимое по горизонтали
            if canvas_width > frame_width:
                x_pos = (canvas_width - frame_width) // 2
            else:
                x_pos = 0

            # Обновляем позицию окна с содержимым
            self.main_canvas.coords(self.main_canvas.find_all()[0], x_pos, 0)

    def on_window_configure(self, event):
        """Обработчик изменения размера окна"""
        # Центрируем содержимое при изменении размера окна
        if event.widget == self.root:
            self.root.after(10, self.center_content)

    def run(self):
        """Запуск приложения"""
        # Финальное центрирование после создания интерфейса
        self.root.after(100, self.center_content)
        self.root.mainloop()

def main():
    """Главная функция"""
    app = BeautifulDigitRecognizer()
    app.run()

if __name__ == "__main__":
    main()
