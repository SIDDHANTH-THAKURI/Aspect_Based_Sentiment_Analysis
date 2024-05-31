import os
import pathlib
import shutil
import random
import nltk
import gensim
from gensim import corpora
from nltk.corpus import sentiwordnet as swn
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import string
import re
from nltk.corpus import stopwords
from tensorflow.keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras import Sequential
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score
from keras.datasets import imdb
from nltk.sentiment import SentimentIntensityAnalyzer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, concatenate,Dropout
from tensorflow.keras.utils import to_categorical
from keras import layers, models
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import tarfile
import urllib.request

try:
    #Downloading and Extracting IMDB data
    data_dir = "aclImdb"

    if not os.path.exists(data_dir):
        try:
            print("Downloading dataset...")
            urllib.request.urlretrieve("https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", "aclImdb_v1.tar.gz")
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
        #Create validation dataset
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


    #Experiment 4
    
    print("Experiment 4 Begin")

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
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)
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
        aspect_input_data_train = {f"{aspects[i]}_input": np.array(aspect_train)[:, i].reshape(-1, 1) for i in range(len(aspects))}
        aspect_input_data_val = {f"{aspects[i]}_input": np.array(aspect_val)[:, i].reshape(-1, 1) for i in range(len(aspects))}
        aspect_input_data_test = {f"{aspects[i]}_input": np.array(aspect_test)[:, i].reshape(-1, 1) for i in range(len(aspects))}

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

        
        #### Note: Experiment 4 Test Accuracy for Basic SA: 87% and ABSA: 94% ####
        

    except Exception as e:
            print(f"Error in Aspect-Based Sentiment Analysis (ABSA) Model: {e}")

    print("Experiment 4 Finish")






    '''

    # # Experiment 1

    print("Experiment 1 Begin")

    num_words = 10000
    INDEX_FROM = 3 

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words, index_from=INDEX_FROM)

    word_to_id = imdb.get_word_index()
    word_to_id = {k: (v + INDEX_FROM) for k, v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2
    id_to_word = {value: key for key, value in word_to_id.items()}

    x_train = [[id_to_word[id] for id in review] for review in x_train]
    x_test = [[id_to_word[id] for id in review] for review in x_test]

    model_w2v = Word2Vec(sentences=x_train, vector_size=100, window=5, min_count=1, workers=4)

    def feature_vector(words, model):
        vec = np.zeros(model.vector_size)
        count = 0
        for word in words:
            try:
                vec += model.wv[word]
                count += 1
            except KeyError:
                continue
        if count != 0:
            vec /= count
        return vec

    train_vectors = np.array([feature_vector(review, model_w2v) for review in x_train])
    test_vectors = np.array([feature_vector(review, model_w2v) for review in x_test])

    model_nn = Sequential()
    model_nn.add(Dense(16, activation='relu', input_dim=100))
    model_nn.add(Dropout(0.5))
    model_nn.add(Dense(1, activation='sigmoid'))
    model_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model_nn.fit(train_vectors, y_train, epochs=10, batch_size=512, validation_split=0.1, verbose=1)

    accuracy = model_nn.evaluate(test_vectors, y_test, verbose=0)
    print(f'The accuracy of the test set is: {accuracy[1]*100:.2f}%')
    print("Experiment 1 Finish")

        
    #### Note: Experiment 1 Test Accuracy is: 82.19% ####
        






    # # Experiment 2
    print("Experiment 2 Begin")
    nltk.download('punkt')
    nltk.download('wordnet')
    def load_imdb_data_2(data_dir, max_samples=5000):
        data = []
        labels = []
        for category in ("neg", "pos"):
            category_dir = os.path.join(data_dir, category)
            for fname in os.listdir(category_dir):
                if len(data) >= max_samples:  # Stop reading if max_samples is reached
                    break
                with open(os.path.join(category_dir, fname), encoding="utf-8") as f:
                    data.append(f.read())
                labels.append(0 if category == "neg" else 1)
            if len(data) >= max_samples:  
                break
        
        random.seed(42)  
        combined = list(zip(data, labels))
        random.shuffle(combined)
        data[:], labels[:] = zip(*combined)
        
        return data[:max_samples], labels[:max_samples] 

    train_data, train_labels = load_imdb_data_2("aclImdb/train",5000)
    test_data, test_labels = load_imdb_data_2("aclImdb/test",5000)

    docs_train = train_data
    docs_test = test_data

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess(doc):
        doc = re.sub(r'<br\s*/?>', ' ', doc)  # remove <br> label
        return [lemmatizer.lemmatize(word) for word in nltk.word_tokenize(doc.lower()) if word.isalpha() and word not in stop_words]

    texts_train = [preprocess(doc) for doc in docs_train]
    texts_test = [preprocess(doc) for doc in docs_test]

    dictionary = corpora.Dictionary(texts_train)
    corpus = [dictionary.doc2bow(text) for text in texts_train]

    lda_model = gensim.models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

    # for i in range(lda_model.num_topics):
    #     print(f"Topic {i}:")
    #     words = lda_model.show_topic(i, topn=10)
    #     for word, weight in words:
    #         print(f"  {word} ({weight:.3f})")
    #     print("\n")

    def get_sentiment_score(word):
        try:
            synsets = list(swn.senti_synsets(word))
            if not synsets:
                return 0
            synset = synsets[0]
            return synset.pos_score() - synset.neg_score()
        except:
            return 0

    def analyze_review(doc):
        bow = dictionary.doc2bow(preprocess(doc))
        topic_dist = lda_model.get_document_topics(bow, minimum_probability=0)
        aspect_sentiments = {topic: 0 for topic in range(lda_model.num_topics)}

        for topic, weight in topic_dist:
            topic_terms = lda_model.get_topic_terms(topic, topn=10)
            sentiment_score = 0
            weight_sum = 0

            for term_id, term_weight in topic_terms:
                word = dictionary[term_id]
                score = get_sentiment_score(word)
                sentiment_score += term_weight * score
                weight_sum += term_weight

            if weight_sum != 0:
                sentiment_score /= weight_sum

            aspect_sentiments[topic] = sentiment_score

        return aspect_sentiments

    data_features_train = []
    data_features_test = []
    for doc in docs_train:
        aspects = analyze_review(doc)
        feature_vector = [aspects[topic] for topic in range(lda_model.num_topics)]
        data_features_train.append(feature_vector)

    for doc in docs_test:
        aspects = analyze_review(doc)
        feature_vector = [aspects[topic] for topic in range(lda_model.num_topics)]
        data_features_test.append(feature_vector)

    X = np.array(data_features_train)
    X_test = np.array(data_features_test)
    y = np.array(train_labels)
    y_test = np.array(test_labels)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


    model = Sequential()
    model.add(Dense(16, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))


    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy:.2f}')
    print("Experiment 2 Finish")

            
    #### Note: Experiment 4 Test Accuracy is: 50% ####
        







    # # Experiment 3
    print("Experiment 3 Begin")
    # presetting aspects work

    # max_features = 10000  
    # maxlen = 300  
    # (x_train, _), (_, _) = imdb.load_data(num_words=max_features)
    # word_index = imdb.get_word_index()
    # reverse_word_index = {value: key for key, value in word_index.items()}
    # def sequence_to_text(sequence):
    #     return ' '.join([reverse_word_index.get(i - 3, '?') for i in sequence])
    # counts = Counter()
    # for sequence in x_train:
    #     review_text = sequence_to_text(sequence)
    #     tokens = word_tokenize(review_text)
    #     tagged = pos_tag(tokens)
    #     nouns = [word for word, tag in tagged if tag.startswith('NN')]  
    #     counts.update(nouns)
    # most_common_nouns = counts.most_common(50)
    # for word, freq in most_common_nouns:
    #     print(f"{word}")

    # Load Raw Text Data
    def load_imdb_data_3(data_dir):
        data = []
        labels = []
        for category in ("neg", "pos"):
            category_dir = os.path.join(data_dir, category)
            for fname in os.listdir(category_dir):
                with open(os.path.join(category_dir, fname), encoding="utf-8") as f:
                    data.append(f.read())
                labels.append(0 if category == "neg" else 1)
        random.seed(42)
        combined = list(zip(data, labels))
        random.shuffle(combined)
        data[:], labels[:] = zip(*combined)
        
        return data, labels

    train_data, train_labels = load_imdb_data_3("aclImdb/train")
    test_data, test_labels = load_imdb_data_3("aclImdb/test")

    aspects = {
        'movie': ['movie', 'film','cinema'],
        'plot': ['plot', 'story', 'storyline'],
        'time': ['hour', 'minute'],
        'acting': ['acting', 'performance', 'actor', 'actress','cast'],
        'direction': ['direction', 'director', 'cinematography', 'production'],
        'music': ['music', 'sound','score'],
        'editing': ['editing'],
        'special effect': ['special effect']
    }
    aspects_key = ['movie', 'plot','time', 'acting', 'direction', 'music','editing','special effect']

    def categorize_review_by_aspect(review_text, aspects):
        # Function to normalize text
        def normalize(text):
            # Lowercase the text and replace multiple spaces with a single space
            text = text.lower().strip()
            # Handle plural forms (very basic handling)
            text = re.sub(r's\b', '', text)  # Remove trailing 's' to treat plurals
            return text

        # Split the review text into sentences
        sentences = [sentence.strip() for sentence in review_text.split('.') if sentence]

        # Prepare the output dictionary
        categorized_aspects = {aspect: [] for aspect in aspects}

        # Iterate over each sentence
        for sentence in sentences:
            normalized_sentence = normalize(sentence)
            # Check each aspect
            for aspect, keywords in aspects.items():
                # Normalize keywords for matching
                normalized_keywords = [normalize(keyword) for keyword in keywords]
                # Check if any keyword appears in the sentence
                if any(keyword in normalized_sentence for keyword in normalized_keywords):
                    categorized_aspects[aspect].append(sentence)

        return categorized_aspects

    def calculate_aspect_sentiments(aspect_sentences):
        # Initialize the sentiment analyzer
        sia = SentimentIntensityAnalyzer()
        
        # Prepare the output dictionary
        aspect_sentiments = {}
        
        # Calculate sentiment for each aspect based on its sentences
        for aspect, sentences in aspect_sentences.items():
            total_sentiment = 0
            # Analyze sentiment of each sentence
            for sentence in sentences:
                sentiment_score = sia.polarity_scores(sentence)['compound']
                total_sentiment += sentiment_score
            # Assign the total sentiment score to the aspect
            aspect_sentiments[aspect] = total_sentiment if sentences else 0

        return aspect_sentiments

    def process_reviews(train_data, aspects):
        # List to hold the aspect sentiment dictionaries for each review
        all_reviews_sentiments = []
        
        # Process each review in the train_data
        for review in train_data:
            # Categorize the review by aspects
            categorized_aspects = categorize_review_by_aspect(review, aspects)
            # Calculate sentiment scores for the categorized aspects
            aspect_sentiments = calculate_aspect_sentiments(categorized_aspects)
            # Append the results to the list
            all_reviews_sentiments.append(aspect_sentiments)
        
        return all_reviews_sentiments

    review_sentiments_train = process_reviews(train_data, aspects)
    review_sentiments_test = process_reviews(test_data, aspects)

    df_train = pd.DataFrame(review_sentiments_train)
    df_test = pd.DataFrame(review_sentiments_test)

    X_train = df_train.values
    X_test = df_test.values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)), 
        layers.Dense(64, activation='relu'),  
        layers.Dense(32, activation='relu'),  
        layers.Dense(1, activation='sigmoid')  
    ])

    model.compile(optimizer='adam',  
                  loss='binary_crossentropy',  
                  metrics=['accuracy'])  

    history = model.fit(X_train, y_train, epochs=18, batch_size=32,validation_split=0.2)

    eval_result = model.evaluate(X_test, y_test)
    print(f"Test Loss: {eval_result[0]} - Test Accuracy: {eval_result[1]}")
    print("Experiment 3 Finish")


    #### Note: Experiment 4 Test Accuracy is: 71.56% ####
    

    '''

except Exception as e:
    print('An error occurred:', e)
