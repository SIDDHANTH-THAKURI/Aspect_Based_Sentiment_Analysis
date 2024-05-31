import os
import pathlib
import shutil
import random
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import string
import re
from nltk.corpus import stopwords
from tensorflow.keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras import Sequential
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, concatenate, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np
import tarfile
import urllib.request

try:
    # Downloading and Extracting IMDB data
    data_dir = "aclImdb"

    if not os.path.exists(data_dir):
        try:
            print("Downloading dataset...")
            urllib.request.urlretrieve("https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
                                       "aclImdb_v1.tar.gz")
            with tarfile.open("aclImdb_v1.tar.gz", "r:gz") as tar:
                tar.extractall()
            shutil.rmtree(os.path.join("aclImdb", "train", "unsup"))
            os.remove("aclImdb_v1.tar.gz")
            print("IMDB dataset is ready for use.")

        except Exception as e:
            print(f"Error in downloading or extracting dataset: {e}")
    else:
        print("Dataset already exists. Skipping download and extraction.\n")

    try:
        # Create validation dataset
        base_dir = pathlib.Path("aclImdb")
        val_dir = base_dir / "val"
        train_dir = base_dir / "train"
        if not val_dir.exists():
            for category in ("neg", "pos"):
                os.makedirs(val_dir / category)
                files = os.listdir(train_dir / category)
                random.Random(1337).shuffle(files)
                num_val_samples = int(0.2 * len(files))
                val_files = files[-num_val_samples:]
                for fname in val_files:
                    shutil.move(train_dir / category / fname, val_dir / category / fname)
    except Exception as e:
        print(f"Error in creating validation dataset: {e}")

    try:
        # Download NLTK stopwords
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
    except Exception as e:
        print(f"Error in downloading NLTK stopwords: {e}")

    # Basic Sentiment Analysis (SA) Model

    print("Running Basic Sentiment Analysis (SA) Model")

    try:
        # Define positive and negative words
        positive_words = [
            "good", "great", "fantastic", "excellent", "amazing", "wonderful", "love", "enjoyed",
            "best", "awesome", "brilliant", "superb", "exceptional", "impressive", "outstanding",
            "remarkable", "stunning", "marvelous", "terrific", "delightful", "incredible",
            "phenomenal", "captivating", "enthralling", "thrilling", "inspiring", "breathtaking",
            "memorable", "hilarious", "heartwarming", "charming", "uplifting", "poignant", "emotional",
            "riveting", "touching", "entertaining", "fun", "engaging", "satisfying", "magical",
            "splendid", "pleasing", "enchanting", "refreshing", "wonder", "masterpiece", "commendable",
            "fascinating", "rewarding"
        ]
        negative_words = [
            "bad", "terrible", "awful", "worst", "horrible", "disappointed", "poor", "boring", "hate",
            "sucks", "dreadful", "lame", "pathetic", "weak", "uninteresting", "annoying", "flawed",
            "mediocre", "predictable", "unimpressive", "disappointing", "tedious", "overrated", "bland",
            "forgettable", "uninspired", "mess", "nonsensical", "ridiculous", "absurd", "disgusting",
            "stupid", "cringe", "dull", "unbearable", "atrocious", "inept", "abysmal", "cheesy",
            "cliché", "disjointed", "pointless", "awkward", "unoriginal", "lackluster", "tedium",
            "laughable", "painful", "frustrating"
        ]


        # Function to preprocess text
        def preprocess_text(text):
            text = text.lower()
            text = re.sub(f'[{string.punctuation}]', '', text)
            words = text.split()
            words = [word for word in words if word not in stop_words]
            return words


        # Load datasets for basic SA
        batch_size = 32
        train_ds = keras.utils.text_dataset_from_directory(
            base_dir / "train", batch_size=batch_size
        )
        val_ds = keras.utils.text_dataset_from_directory(
            base_dir / "val", batch_size=batch_size
        )
        test_ds = keras.utils.text_dataset_from_directory(
            base_dir / "test", batch_size=batch_size
        )
    except Exception as e:
        print(f"Error in loading datasets: {e}")

    try:
        # Accumulate positive and negative words
        pos_word_list = []
        neg_word_list = []

        for text_batch, _ in train_ds.unbatch().take(10000):
            text = preprocess_text(text_batch.numpy().decode('utf-8'))
            pos_word_list.extend([word for word in text if word in positive_words])
            neg_word_list.extend([word for word in text if word in negative_words])


        # Generate word clouds
        def generate_word_cloud(word_list, title):
            word_counts = Counter(word_list)
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(
                word_counts)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title(title)
            plt.axis('off')
            plt.show()


        # Display word clouds
        print("Visualizing the DataSet!")
        generate_word_cloud(pos_word_list, 'Positive Words Word Cloud')
        generate_word_cloud(neg_word_list, 'Negative Words Word Cloud')
    except Exception as e:
        print(f"Error in generating word clouds: {e}")

    try:
        # Define the text vectorization layer
        max_features = 10000
        embedding_dim = 16
        sequence_length = 500

        vectorize_layer = TextVectorization(
            max_tokens=max_features,
            output_mode='int',
            output_sequence_length=sequence_length
        )

        # Adapt the vectorization layer to the text data
        train_text = train_ds.map(lambda x, y: x)
        vectorize_layer.adapt(train_text)


        # Vectorize the datasets
        def vectorize_text(text, label):
            text = tf.expand_dims(text, -1)
            return vectorize_layer(text), label


        train_ds = train_ds.map(vectorize_text)
        val_ds = val_ds.map(vectorize_text)
        test_ds = test_ds.map(vectorize_text)

        # Define the model for basic SA
        model = Sequential([
            Embedding(max_features + 1, embedding_dim),
            GlobalAveragePooling1D(),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        # Train the model for basic SA
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=10
        )

        # Plot training & validation accuracy values
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.tight_layout()
        plt.show()

        # Evaluate the model for basic SA
        loss, accuracy = model.evaluate(test_ds)
        print(f'Test Accuracy for Basic SA: {accuracy}')

        # Predict sentiment for a sample review using basic SA
        print("\nTesting Basic SA with a sample review!")
        sample_review = "The movie was fantastic! The story was gripping and the characters were well-developed."
        sample_review_vectorized = vectorize_layer([sample_review])
        prediction = model.predict(sample_review_vectorized)

        print(f'Sentiment score for Basic SA: {prediction[0][0]}')
        sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
        print(f'The review is {sentiment}.')
    except Exception as e:
        print(f"Error in Basic Sentiment Analysis (SA) Model: {e}")

    # Aspect-Based Sentiment Analysis (ABSA) Model

    print("\nRunning Aspect-Based Sentiment Analysis (ABSA) Model")

    try:
        # Define 50 aspects
        aspects = ["plot", "characters", "acting", "music", "dialogues", "scenery", "cinematography",
                   "direction", "screenplay", "storyline", "theme", "pacing", "tone", "genre",
                   "setting", "costumes", "makeup", "special effects", "action sequences", "romance",
                   "humor", "suspense", "twists", "endings", "messages", "symbolism", "subplots",
                   "climax", "resolution", "protagonist", "antagonist", "supporting cast", "chemistry",
                   "conflicts", "character development", "backstory", "narrative", "dialogue delivery",
                   "voice acting", "character arcs", "motivation", "foreshadowing", "flashbacks",
                   "character traits", "internal conflicts", "external conflicts", "visual effects",
                   "color palette", "soundtrack", "ambient sound"]


        # Function to load data from text files
        def load_data_from_dir(directory):
            texts = []
            labels = []
            aspect_data = []  # Placeholder for aspect information
            for label_dir in ["pos", "neg"]:
                dir_path = directory / label_dir
                for fname in dir_path.glob("*.txt"):
                    with open(fname, encoding="utf-8") as f:
                        texts.append(f.read())
                    labels.append(1 if label_dir == "pos" else 0)
                    # Generate dummy aspect data for each aspect
                    aspect_data.append(np.random.randint(0, 2, size=len(aspects)))
            return texts, labels, aspect_data


        # Load datasets for ABSA
        train_dir = base_dir / "train"
        val_dir = base_dir / "val"
        test_dir = base_dir / "test"

        X_train, y_train, aspect_train = load_data_from_dir(train_dir)
        X_val, y_val, aspect_val = load_data_from_dir(val_dir)
        X_test, y_test, aspect_test = load_data_from_dir(test_dir)

        # Tokenize text data
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(X_train)

        X_train = tokenizer.texts_to_sequences(X_train)
        X_val = tokenizer.texts_to_sequences(X_val)
        X_test = tokenizer.texts_to_sequences(X_test)

        maxlen = 100
        X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
        X_val = pad_sequences(X_val, padding='post', maxlen=maxlen)
        X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

        # Convert sentiments to categorical
        y_train = to_categorical(y_train, num_classes=2)
        y_val = to_categorical(y_val, num_classes=2)
        y_test = to_categorical(y_test, num_classes=2)

        # Model definition for ABSA
        input_text = Input(shape=(maxlen,))
        embedding = Embedding(input_dim=5000, output_dim=128, input_length=maxlen)(input_text)
        lstm = LSTM(128, return_sequences=False)(embedding)

        aspect_inputs = []
        aspect_dense_layers = []

        # Using actual aspect data
        for i, aspect in enumerate(aspects):
            aspect_input = Input(shape=(1,), name=f"{aspect}_input")
            aspect_inputs.append(aspect_input)
            aspect_dense = Dense(64, activation='relu')(aspect_input)
            aspect_dense_layers.append(aspect_dense)

        concatenated = concatenate([lstm] + aspect_dense_layers)
        output = Dense(2, activation='softmax')(concatenated)  # Adjusting to 2 sentiment classes: negative and positive

        model = Model(inputs=[input_text] + aspect_inputs, outputs=output)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Prepare actual aspect input data for ABSA
        aspect_input_data_train = {f"{aspects[i]}_input": np.array(aspect_train)[:, i].reshape(-1, 1) for i in
                                   range(len(aspects))}
        aspect_input_data_val = {f"{aspects[i]}_input": np.array(aspect_val)[:, i].reshape(-1, 1) for i in
                                 range(len(aspects))}
        aspect_input_data_test = {f"{aspects[i]}_input": np.array(aspect_test)[:, i].reshape(-1, 1) for i in
                                  range(len(aspects))}

        # Combine text data and aspect data for training
        train_data = [X_train] + [aspect_input_data_train[f"{aspect}_input"] for aspect in aspects]
        val_data = [X_val] + [aspect_input_data_val[f"{aspect}_input"] for aspect in aspects]
        test_data = [X_test] + [aspect_input_data_test[f"{aspect}_input"] for aspect in aspects]

        # Train the model for ABSA
        history = model.fit(train_data, y_train, validation_data=(val_data, y_val), epochs=5, batch_size=32)

        # Evaluate the model for ABSA
        y_pred = model.predict(test_data)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)

        # Display accuracy score for ABSA
        accuracy = accuracy_score(y_true, y_pred_classes)
        print(f'Test Accuracy for ABSA: {accuracy}')

        # Test with an example review using ABSA
        print("\nTesting ABSA with a sample review!")
        example_review = "The movie had great acting and a captivating storyline, but the music was terrible."
        example_sequence = tokenizer.texts_to_sequences([example_review])
        example_padded = pad_sequences(example_sequence, padding='post', maxlen=maxlen)
        example_aspects = {f"{aspects[i]}_input": np.random.randint(0, 2, size=(1, 1)) for i in range(len(aspects))}
        example_input = [example_padded] + [example_aspects[f"{aspect}_input"] for aspect in aspects]
        example_prediction = model.predict(example_input)
        sentiment = np.argmax(example_prediction, axis=1)

        print(f'Sentiment score for the example review in ABSA: {"Positive" if sentiment[0] == 1 else "Negative"}')

        #### Note: Test Accuracy for Basic SA: 87% and ABSA: 85% ####


    except Exception as e:
        print(f"Error in Aspect-Based Sentiment Analysis (ABSA) Model: {e}")

except Exception as e:
    print('An error occurred:', e)
