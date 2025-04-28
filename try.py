import numpy as np
import pandas as pd
import re
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Библиотеки для обработки русского текста
import nltk

nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymorphy3

# Библиотеки для нейронной сети
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Загрузка необходимых ресурсов NLTK
nltk.download('stopwords')

# Инициализация морфологического анализатора для русского языка
morph = pymorphy3.MorphAnalyzer()

# Словарь МКБ-10 кодов и соответствующих заболеваний глаз
eye_diseases = {
    "H00": "Гордеолум (ячмень) и халазион",
    "H01": "Другие воспаления век (блефарит)",
    "H02": "Другие болезни век (энтропион, эктропион, лагофтальм)",
    "H04": "Болезни слезного аппарата (сухой глаз, дакриоаденит)",
    "H05": "Болезни глазницы (экзофтальм, энофтальм)",
    "H10": "Конъюнктивит (бактериальный, вирусный, аллергический)",
    "H11": "Другие болезни конъюнктивы (птеригий, пингвекула)",
    "H15": "Болезни склеры (склерит, эписклерит)",
    "H16": "Кератит (воспаление роговицы)",
    "H20": "Иридоциклит (увеит)",
    "H25": "Старческая катаракта",
    "H26": "Другие катаракты",
    "H30": "Хориоретинальное воспаление",
    "H33": "Отслойка и разрывы сетчатки",
    "H35": "Другие болезни сетчатки (дегенерация макулы)",
    "H40": "Глаукома",
    "H43": "Болезни стекловидного тела (помутнение, кровоизлияние)",
    "H46": "Неврит зрительного нерва",
    "H52": "Нарушения рефракции и аккомодации (миопия, гиперметропия, астигматизм)",
    "H53": "Зрительные расстройства (амблиопия, диплопия)",
    "H54": "Слепота и пониженное зрение"
}

# Создание симптоматического словаря для каждого заболевания
symptoms_dict = {
    "H00": ["ячмень", "шишка на веке", "болезненный бугорок", "гной", "отек века", "покраснение века", "халазион",
            "уплотнение на веке"],
    "H01": ["блефарит", "воспаление века", "зуд век", "покраснение краев век", "корки на ресницах", "жжение век",
            "шелушение век"],
    "H04": ["сухость глаз", "песок в глазах", "недостаток слез", "избыток слез", "слезотечение",
            "воспаление слезной железы"],
    "H05": ["выпячивание глаза", "западание глаза", "боль в глазнице", "ограничение движения глаза", "экзофтальм",
            "энофтальм"],
    "H10": ["конъюнктивит", "красный глаз", "гнойные выделения", "слизистые выделения", "зуд в глазу", "жжение в глазу",
            "слезотечение"],
    "H15": ["склерит", "эписклерит", "боль в глазу", "покраснение белка", "глубокая боль"],
    "H16": ["кератит", "боль в глазу", "светобоязнь", "слезотечение", "ощущение инородного тела",
            "помутнение роговицы"],
    "H20": ["иридоциклит", "увеит", "боль в глазу", "светобоязнь", "покраснение глаза", "изменение формы зрачка"],
    "H25": ["катаракта", "помутнение зрения", "ореолы вокруг источников света", "затуманивание зрения",
            "снижение зрения"],
    "H33": ["отслойка сетчатки", "разрыв сетчатки", "вспышки света", "плавающие мушки", "завеса перед глазами",
            "потеря бокового зрения"],
    "H35": ["дегенерация макулы", "искажение изображения", "снижение центрального зрения", "сложности при чтении",
            "ретинопатия"],
    "H40": ["глаукома", "повышенное внутриглазное давление", "потеря периферического зрения", "боль в глазу",
            "головная боль"],
    "H43": ["помутнение стекловидного тела", "кровоизлияние в стекловидное тело", "плавающие помутнения",
            "мушки перед глазами"],
    "H46": ["неврит зрительного нерва", "внезапное снижение зрения", "боль при движении глаза",
            "нарушение цветовосприятия"],
    "H52": ["близорукость", "дальнозоркость", "астигматизм", "нечеткое зрение", "напряжение глаз", "головная боль"],
    "H53": ["амблиопия", "диплопия", "скотома", "сниженное сумеречное зрение", "двоение в глазах",
            "туман перед глазами"],
    "H54": ["слепота", "слабовидение", "потеря зрения", "частичная потеря зрения", "снижение зрения"]
}

# Словарь синонимов для симптомов
symptoms_synonyms = {
    "красный глаз": ["покраснение глаза", "глаз красный", "глаз покраснел"],
    "слезотечение": ["слезы текут", "слезиться", "глаза слезятся"],
    "сухость глаз": ["сухие глаза", "ощущение сухости в глазах"],
    "мушки перед глазами": ["плавающие точки", "плавающие мушки", "мушки в глазах"],
    "светобоязнь": ["боязнь света", "чувствительность к свету", "дискомфорт от яркого света"],
    "двоение": ["диплопия", "двоится в глазах", "двойное зрение"],
    "ячмень": ["гордеолум", "воспаление века", "гнойник на веке"],
    "катаракта": ["помутнение хрусталика", "серая пелена перед глазами"],
    "глаукома": ["повышенное внутриглазное давление", "повышенное глазное давление"]
}

# Веса симптомов по важности для диагностики
symptom_weights = {
    "сильная боль": 2.0,
    "внезапное снижение зрения": 2.5,
    "резкая боль": 2.0,
    "сильное покраснение": 1.5,
    "вспышки света": 2.0,
    "потеря зрения": 2.5,
    "глаукома": 2.0,
    "катаракта": 1.8,
    "отслойка сетчатки": 2.5
}


class EnhancedTextProcessor:
    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()
        self.stop_words = set(stopwords.words('russian'))
        # Дополнительные стоп-слова для медицинского контекста
        self.medical_stop_words = {'это', 'было', 'меня', 'очень', 'как', 'когда', 'если', 'при', 'с', 'в', 'на', 'у'}
        self.stop_words.update(self.medical_stop_words)

        # Словари и связи
        self.symptoms_synonyms = symptoms_synonyms
        self.symptom_weights = symptom_weights

        # Создаем инвертированный словарь синонимов для быстрого поиска
        self.inverted_synonyms = {}
        for main_symptom, synonyms in self.symptoms_synonyms.items():
            for synonym in synonyms:
                self.inverted_synonyms[synonym] = main_symptom

    def normalize_symptom(self, symptom):
        """Нормализует симптом, заменяя его основной формой из словаря синонимов"""
        return self.inverted_synonyms.get(symptom, symptom)

    def extract_medical_phrases(self, text):
        """Выделяет медицинские фразы из текста"""
        medical_phrases = []

        # Ищем все возможные n-граммы, которые могут быть медицинскими терминами
        words = text.lower().split()

        # Проверяем униграммы (отдельные слова)
        for word in words:
            if word in self.symptom_weights or any(word in symptoms for symptoms in symptoms_dict.values()):
                medical_phrases.append(word)

        # Проверяем биграммы и триграммы (словосочетания из 2-3 слов)
        for i in range(len(words) - 1):
            bigram = " ".join(words[i:i + 2])
            if bigram in self.symptoms_synonyms or bigram in self.inverted_synonyms:
                medical_phrases.append(bigram)

            if i < len(words) - 2:
                trigram = " ".join(words[i:i + 3])
                if trigram in self.symptoms_synonyms or trigram in self.inverted_synonyms:
                    medical_phrases.append(trigram)

        return medical_phrases

    def enrich_with_synonyms(self, tokens):
        """Обогащает набор токенов синонимами симптомов"""
        enriched_tokens = tokens.copy()

        for token in tokens:
            # Проверяем, есть ли токен в основном словаре синонимов
            if token in self.symptoms_synonyms:
                # Добавляем все синонимы с небольшим весом
                enriched_tokens.extend([syn.replace(" ", "_") for syn in self.symptoms_synonyms[token]])

            # Проверяем, есть ли токен в инвертированном словаре
            elif token in self.inverted_synonyms:
                main_symptom = self.inverted_synonyms[token]
                enriched_tokens.append(main_symptom.replace(" ", "_"))

        return enriched_tokens

    def preprocess_text(self, text):
        """
        Улучшенная предобработка текста:
        - приведение к нижнему регистру
        - удаление спецсимволов
        - лемматизация
        - обработка медицинских терминов
        - обогащение синонимами
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)

        # Выделяем медицинские фразы перед токенизацией
        medical_phrases = self.extract_medical_phrases(text)

        # Заменяем пробелы в медицинских фразах на подчеркивания
        for phrase in medical_phrases:
            if ' ' in phrase:
                text = text.replace(phrase, phrase.replace(' ', '_'))

        # Токенизация
        tokens = word_tokenize(text)

        # Лемматизация и фильтрация стоп-слов
        lemmatized_tokens = []
        for token in tokens:
            if token not in self.stop_words:
                # Если это составной термин (с подчеркиваниями), сохраняем как есть
                if '_' in token:
                    lemmatized_tokens.append(token)
                else:
                    lemma = self.morph.parse(token)[0].normal_form
                    lemmatized_tokens.append(lemma)

        # Обогащение синонимами
        enriched_tokens = self.enrich_with_synonyms(lemmatized_tokens)

        return ' '.join(enriched_tokens)

    def generate_synthetic_data(self, symptoms_dict, n_samples_per_class=20):
        """
        Генерация синтетического набора данных для обучения модели.
        Создает разнообразные комбинации симптомов для каждого заболевания.
        """
        data = []

        for code, symptoms_list in symptoms_dict.items():
            for _ in range(n_samples_per_class):
                # Выбираем случайное количество симптомов (от 2 до 5)
                n_symptoms = np.random.randint(2, min(6, len(symptoms_list) + 1))
                selected_symptoms = np.random.choice(symptoms_list, n_symptoms, replace=False)

                # Создаем жалобу пациента
                complaint = "У меня " + ", ".join(selected_symptoms[:-1])
                if len(selected_symptoms) > 1:
                    complaint += " и " + selected_symptoms[-1]
                else:
                    complaint = "У меня " + selected_symptoms[0]

                # Добавляем вариативность в текст жалобы
                modifiers = [
                    "сильно ", "очень ", "немного ", "слегка ", "постоянно ",
                    "иногда ", "периодически ", "в последнее время "
                ]

                # Дополнительные фразы для контекста
                context_phrases = [
                    "Я заметил это ", "Это началось ", "Это происходит ",
                    "Я обратил внимание ", "Это беспокоит меня "
                ]

                # Временные указатели
                time_phrases = [
                    "вчера", "сегодня утром", "неделю назад",
                    "несколько дней назад", "давно", "внезапно",
                    "постепенно", "последние дни"
                ]

                # Применяем модификаторы к симптомам
                complaint_words = complaint.split()
                for i in range(len(complaint_words)):
                    if np.random.random() < 0.3:  # 30% шанс модификации
                        if i > 0 and complaint_words[i] in ["красный", "болит", "зудит", "воспаленный", "опухший",
                                                            "сухой"]:
                            complaint_words[i] = np.random.choice(modifiers) + complaint_words[i]

                complaint = " ".join(complaint_words)

                # Добавляем контекст (с 40% вероятностью)
                if np.random.random() < 0.4:
                    context = np.random.choice(context_phrases) + np.random.choice(time_phrases) + "."
                    complaint += " " + context

                # Добавляем дополнительные детали (с 30% вероятностью)
                if np.random.random() < 0.3:
                    details = [
                        " Это мешает мне работать.",
                        " Из-за этого трудно читать.",
                        " Это происходит особенно по утрам.",
                        " Это усиливается при ярком свете.",
                        " Это улучшается, когда я закрываю глаза."
                    ]
                    complaint += np.random.choice(details)

                data.append([complaint, code])

                # Создаем дополнительные варианты с синонимами
                if np.random.random() < 0.5:  # 50% шанс
                    for symptom in selected_symptoms:
                        if symptom in symptoms_synonyms:
                            synonyms = symptoms_synonyms[symptom]
                            synonym = np.random.choice(synonyms)
                            alt_complaint = complaint.replace(symptom, synonym)
                            data.append([alt_complaint, code])

        return pd.DataFrame(data, columns=['complaint', 'diagnosis'])


class EnhancedEyeDiseaseClassifier:
    def __init__(self):
        self.text_processor = EnhancedTextProcessor()
        self.vectorizer = None
        self.model = None
        self.label_encoder = None
        self.eye_diseases = eye_diseases
        self.feature_names = None
        self.history = None

    def prepare_data(self, df=None, test_size=0.2, random_state=42):
        """Подготовка данных для обучения модели"""
        # Если датафрейм не передан, генерируем синтетические данные
        if df is None:
            df = self.text_processor.generate_synthetic_data(symptoms_dict)
            print(f"Сгенерировано {len(df)} синтетических примеров жалоб пациентов")

        # Предобработка жалоб
        print("Предобработка текстовых данных...")
        df['processed_complaint'] = df['complaint'].apply(self.text_processor.preprocess_text)

        # Векторизация текста с помощью TF-IDF с улучшенными параметрами
        print("Векторизация текста...")
        self.vectorizer = TfidfVectorizer(
            max_features=2000,  # Увеличиваем количество признаков
            ngram_range=(1, 3),  # Учитываем униграммы, биграммы и триграммы
            min_df=3,  # Минимальная частота встречаемости
            use_idf=True,  # Используем IDF
            norm='l2'  # Нормализация
        )
        X = self.vectorizer.fit_transform(df['processed_complaint'])
        self.feature_names = self.vectorizer.get_feature_names_out()

        # Кодирование меток (диагнозов)
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df['diagnosis'])

        # Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"Размеры выборок: Обучающая - {X_train.shape}, Тестовая - {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def build_model(self, input_shape, num_classes):
        """Создание улучшенной модели нейронной сети"""
        model = Sequential([
            # Входной слой
            Dense(256, activation='relu', input_shape=(input_shape,)),
            BatchNormalization(),
            Dropout(0.4),

            # Скрытые слои
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),

            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),

            # Выходной слой
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=['accuracy']
        )

        # Вывод информации о модели
        model.summary()

        return model

    def train(self, epochs=50, batch_size=32, patience=10, model_path='eye_disease_model.h5'):
        """Обучение модели с дополнительными улучшениями"""
        # Подготовка данных
        X_train, X_test, y_train, y_test = self.prepare_data()

        # Создание модели
        print("Создание и компиляция модели...")
        self.model = self.build_model(X_train.shape[1], len(self.label_encoder.classes_))

        # Настройка колбэков
        callbacks = [
            # Ранняя остановка
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),

            # Сохранение лучшей модели
            ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),

            # Динамическое изменение скорости обучения
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001,
                verbose=1
            )
        ]

        # Обучение модели
        print(f"Начало обучения модели на {epochs} эпохах...")
        self.history = self.model.fit(
            X_train.toarray(), y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )

        # Оценка модели
        loss, accuracy = self.model.evaluate(X_test.toarray(), y_test)
        print(f"Точность на тестовой выборке: {accuracy:.4f}")

        # Предсказания на тестовой выборке
        y_pred = self.model.predict(X_test.toarray())
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Преобразование обратно в коды МКБ-10
        y_pred_labels = self.label_encoder.inverse_transform(y_pred_classes)
        y_test_labels = self.label_encoder.inverse_transform(y_test)

        # Отчет о классификации
        print("\nОтчет о классификации:")
        print(classification_report(y_test_labels, y_pred_labels))

        # Визуализация процесса обучения
        self.plot_training_history()

        return self.history

    def plot_training_history(self):
        """Визуализация процесса обучения модели"""
        if self.history is None:
            print("История обучения недоступна.")
            return

        plt.figure(figsize=(12, 5))

        # График точности
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Обучающая выборка')
        plt.plot(self.history.history['val_accuracy'], label='Проверочная выборка')
        plt.title('Точность модели')
        plt.ylabel('Точность')
        plt.xlabel('Эпоха')
        plt.legend()

        # График функции потерь
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Обучающая выборка')
        plt.plot(self.history.history['val_loss'], label='Проверочная выборка')
        plt.title('Функция потерь')
        plt.ylabel('Потери')
        plt.xlabel('Эпоха')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()

    def save_model(self, model_path='eye_disease_model.h5', vectorizer_path='vectorizer.pkl',
                   encoder_path='label_encoder.pkl'):
        """Сохранение модели и всех необходимых компонентов"""
        if self.model is None:
            raise ValueError("Модель не обучена. Запустите метод train() перед сохранением.")

        self.model.save(model_path)
        print(f"Модель сохранена в {model_path}")

        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"Векторизатор сохранен в {vectorizer_path}")

        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"Кодировщик меток сохранен в {encoder_path}")

        with open('eye_diseases.json', 'w', encoding='utf-8') as f:
            json.dump(self.eye_diseases, f, ensure_ascii=False, indent=4)
        print(f"Словарь заболеваний сохранен в eye_diseases.json")

        print("Все компоненты успешно сохранены.")

    def load_model(self, model_path='eye_disease_model.h5', vectorizer_path='vectorizer.pkl',
                   encoder_path='label_encoder.pkl', diseases_path='eye_diseases.json'):
        """Загрузка модели и всех необходимых компонентов"""
        try:
            self.model = load_model(model_path)
            print(f"Модель загружена из {model_path}")

            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            print(f"Векторизатор загружен из {vectorizer_path}")

            self.feature_names = self.vectorizer.get_feature_names_out()

            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print(f"Кодировщик меток загружен из {encoder_path}")

            with open(diseases_path, 'r', encoding='utf-8') as f:
                self.eye_diseases = json.load(f)
            print(f"Словарь заболеваний загружен из {diseases_path}")

            print("Все компоненты успешно загружены.")
            return True
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            return False

    def get_important_features(self, processed_text, top_n=10):
        """Определение наиболее важных признаков для конкретного текста"""
        if self.vectorizer is None or self.feature_names is None:
            raise ValueError("Векторизатор не инициализирован")

        # Получаем вектор TF-IDF для текста
        text_vector = self.vectorizer.transform([processed_text])

        # Получаем индексы ненулевых элементов
        nonzero_indices = text_vector.nonzero()[1]

        # Создаем список пар (признак, значение)
        feature_values = [(self.feature_names[i], text_vector[0, i]) for i in nonzero_indices]

        # Сортируем по убыванию значения и берем top_n
        top_features = sorted(feature_values, key=lambda x: x[1], reverse=True)[:top_n]

        return top_features

    def predict(self, complaint_text, top_n=3):
        """
        Улучшенное предсказание заболевания на основе жалобы пациента
        с объяснением результатов и выделением ключевых симптомов
        """
        # Проверка, что модель загружена
        if self.model is None:
            raise ValueError("Модель не загружена. Используйте метод train() или load_model()")

        # Предобработка входного текста
        processed_text = self.text_processor.preprocess_text(complaint_text)

        # Получение важных признаков
        important_features = self.get_important_features(processed_text)

        # Преобразование текста в TF-IDF вектор
        text_vector = self.vectorizer.transform([processed_text])

        # Предсказание
        prediction = self.model.predict(text_vector.toarray())[0]

        # Получение индексов наиболее вероятных диагнозов
        top_indices = np.argsort(prediction)[::-1][:top_n]

        # Формирование результата
        top_diagnoses = []
        for idx in top_indices:
            code = self.label_encoder.inverse_transform([idx])[0]
            disease = self.eye_diseases.get(code, "Неизвестное заболевание")
            prob = float(prediction[idx])

            # Получаем типичные симптомы для этого заболевания
            typical_symptoms = symptoms_dict.get(code, [])

            top_diagnoses.append({
                'code': code,
                'disease': disease,
                'probability': prob,
                'typical_symptoms': typical_symptoms
            })

        # Возвращаем результат с объяснением
        return {
            'main_diagnosis': top_diagnoses[0],
            'alternative_diagnoses': top_diagnoses[1:],
            'important_features': important_features
        }

    def explain_diagnosis(self, result):
        """Генерирует текстовое объяснение диагноза на основе результатов предсказания"""
        main_diag = result['main_diagnosis']
        important_features = result['important_features']

        explanation = f"Наиболее вероятный диагноз: {main_diag['code']} - {main_diag['disease']} (вероятность: {main_diag['probability']:.2%}).\n\n"

        # Объяснение на основе ключевых симптомов
        explanation += "Этот диагноз основан на следующих ключевых симптомах:\n"
        for feature, value in important_features[:5]:
            # Убираем подчеркивания, если они есть (были добавлены при обработке)
            feature_name = feature.replace('_', ' ')
            explanation += f"- {feature_name}\n"

        # Типичные симптомы
        explanation += f"\nТипичные симптомы для {main_diag['code']} включают:\n"
        for symptom in main_diag['typical_symptoms'][:5]:
            explanation += f"- {symptom}\n"

        # Альтернативные диагнозы
        if result['alternative_diagnoses']:
            explanation += "\nАльтернативные диагнозы для рассмотрения:\n"
            for diag in result['alternative_diagnoses']:
                explanation += f"- {diag['code']} - {diag['disease']} (вероятность: {diag['probability']:.2%})\n"

        # Рекомендация
        explanation += "\nВАЖНО: Это предварительный диагноз, основанный только на текстовом описании симптомов. "
        explanation += "Для точного диагноза необходимо обратиться к офтальмологу для полного обследования."

        return explanation


def main():
    # Создание экземпляра классификатора
    classifier = EnhancedEyeDiseaseClassifier()

    # Проверка наличия сохраненной модели
    print("Проверка наличия сохраненной модели...")
    try:
        loaded = classifier.load_model()
        if not loaded:
            raise FileNotFoundError("Модель не найдена или повреждена")
    except Exception as e:
        print(f"Сохраненная модель не найдена: {e}. Приступаем к обучению новой модели.")

        # Обучение модели
        print("Начинаем обучение модели...")
        classifier.train(epochs=50, batch_size=32)

        # Сохранение модели
        print("Сохранение обученной модели...")
        classifier.save_model()

    # Примеры предсказаний
    example_complaints = [
        "У меня сильно красный глаз, гнойные выделения и зуд",
        "Я вижу плавающие мушки перед глазами и иногда вспышки света",
        "У меня на веке появилась небольшая шишка, болезненная при надавливании",
        "Глаза сухие, ощущение песка в глазах, слезятся на ветру",
        "Постепенно ухудшается зрение, особенно в сумерках, вижу ореолы вокруг фонарей"
    ]

    print("\n--- Примеры предсказаний ---")
    for i, complaint in enumerate(example_complaints, 1):
        result = classifier.predict(complaint)
        explanation = classifier.explain_diagnosis(result)

        print(f"\nПример {i}:")
        print(f"Жалоба пациента: {complaint}")
        print("=" * 50)
        print(explanation)
        print("=" * 50)

    # Интерактивный режим
    print("\n--- Интерактивный режим ---")
    print("Введите жалобу пациента (или 'выход' для завершения):")

    while True:
        user_input = input("\nЖалоба: ")
        if user_input.lower() in ['выход', 'exit', 'quit']:
            break

        result = classifier.predict(user_input)
        explanation = classifier.explain_diagnosis(result)

        print("\n" + "=" * 50)
        print(explanation)
        print("=" * 50)


if __name__ == "__main__":
    main()
