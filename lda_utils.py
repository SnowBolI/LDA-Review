import os
import re
import json
import pickle
import pandas as pd
import nltk
from gensim import corpora, models
from progress_utils import update_progress
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import pyLDAvis.gensim_models
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# === NLTK setup dengan auto-download ===
try:
    # Coba akses stopwords, jika gagal maka download
    stop_words = set(stopwords.words('english'))
except LookupError:
    # Download data yang diperlukan
    print("Downloading NLTK data...")
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    stop_words = set(stopwords.words('english'))

custom_stopwords = {'like', 'ok', 'im', 'get', 'one', 'really', 'app'}
stop_words.update(custom_stopwords)

lemmatizer = WordNetLemmatizer()

def scraper_progress_callback(current, total, app_name):
    """Callback function for scraper progress"""
    if total > 0:
        scrape_percent = (current / total) * 8  # 8% of total progress for scraping
        update_progress(2 + scrape_percent, f"Scraping {app_name}: {current}/{total} reviews", app_name)

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def check_cancel(app_name=None):
    """Check if training was cancelled for specific app"""
    if app_name:
        cancel_file = f"cancel_{app_name}.flag"
    else:
        cancel_file = "cancel.flag"
    return os.path.exists(cancel_file)

# === App IDs ===
APP_IDS = {
    'honkai-star-rail': 'com.HoYoverse.hkrpgoversea',
    'amazon': 'com.amazon.mShop.android.shopping',
    'wuthering-waves': 'com.kurogame.wutheringwaves.global',
    'arena-breakout': 'com.proximabeta.mf.uamo',
    'spotify': 'com.spotify.music'
}

# === Data Loading ===
# Add this function to lda_utils.py for debugging
def debug_data_structure(app_name):
    """Debug function to check data structure"""
    path = f"data/data_per_app/{app_name}.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"[DEBUG] File {app_name}.csv exists")
        print(f"[DEBUG] Shape: {df.shape}")
        print(f"[DEBUG] Columns: {df.columns.tolist()}")
        print(f"[DEBUG] First few rows:")
        print(df.head())
        
        # Check for common content columns
        content_columns = []
        for col in df.columns:
            if col.lower() in ['content', 'review', 'text', 'comment', 'body', 'message']:
                content_columns.append(col)
        
        print(f"[DEBUG] Potential content columns: {content_columns}")
        
        if content_columns:
            sample_col = content_columns[0]
            print(f"[DEBUG] Sample data from '{sample_col}':")
            print(df[sample_col].head(3).tolist())
        
        return df
    else:
        print(f"[DEBUG] File {path} does not exist")
        return None

# Modified load_data function with better error handling
def load_data(app_name):
    path = f"data/data_per_app/{app_name}.csv"
    if not os.path.exists(path):
        update_progress(2, f"Data tidak ditemukan, mulai scraping {app_name}...", app_name)
        try:
            from scraper import scrape_reviews
            df = scrape_reviews(app_name)
            if df is None or len(df) == 0:
                raise ValueError(f"Scraping gagal atau tidak ada data untuk {app_name}")
            
            update_progress(8, f"Scraping selesai, menyimpan data {app_name}...", app_name)
            os.makedirs("data/data_per_app", exist_ok=True)
            df.to_csv(path, index=False)
            update_progress(10, f"Data {app_name} berhasil disimpan", app_name)
        except Exception as e:
            error_msg = f"Scraping gagal: {str(e)}"
            print(f"[LDA] {error_msg}")
            update_progress(0, error_msg, app_name)
            raise ValueError(error_msg)
    else:
        update_progress(2, f"Memuat data {app_name} dari file...", app_name)
        update_progress(5, f"Membaca file {app_name}...", app_name)
        try:
            df = pd.read_csv(path)
            if len(df) == 0:
                raise ValueError(f"File {app_name}.csv kosong")
            update_progress(10, f"Data {app_name} berhasil dimuat ({len(df)} reviews)", app_name)
        except Exception as e:
            error_msg = f"Gagal membaca file {app_name}.csv: {str(e)}"
            print(f"[LDA] {error_msg}")
            update_progress(0, error_msg, app_name)
            raise ValueError(error_msg)
    
    # Debug the loaded data
    print(f"[DEBUG] Loaded data for {app_name}:")
    debug_data_structure(app_name)
    
    return df

# === Preprocessing ===
def preprocess_data(df):
    try:
        def preprocess(text):
            if pd.isna(text):
                return []
            text = str(text).lower()
            text = re.sub(r'[^a-z\s]', '', text)
            tokens = text.split()
            filtered = [t for t in tokens if t not in stop_words and len(t) > 2]
            if not filtered:
                return []
            pos_tags = pos_tag(filtered)
            lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
            return lemmatized

        print(f"Preprocessing {len(df)} documents...")
        df['tokens'] = df['content'].apply(preprocess)
        
        # Filter out empty token lists
        df = df[df['tokens'].apply(len) > 0].reset_index(drop=True)
        print(f"After filtering: {len(df)} documents with valid tokens")

        bigram = Phrases(df['tokens'], min_count=5, threshold=10)
        bigram_mod = Phraser(bigram)
        trigram = Phrases(bigram_mod[df['tokens']], min_count=5, threshold=15)
        trigram_mod = Phraser(trigram)

        df['bigram_trigram_tokens'] = df['tokens'].apply(lambda x: trigram_mod[bigram_mod[x]])
        
        # Verify the column was created
        if 'bigram_trigram_tokens' not in df.columns:
            raise ValueError("Failed to create bigram_trigram_tokens column")

        dictionary = corpora.Dictionary(df['bigram_trigram_tokens'])
        dictionary.filter_extremes(no_below=5, no_above=0.5)
        corpus = [dictionary.doc2bow(text) for text in df['bigram_trigram_tokens']]

        print(f"Dictionary size: {len(dictionary)}")
        print(f"Corpus size: {len(corpus)}")

        return df, dictionary, corpus
        
    except Exception as e:
        print(f"Preprocessing error: {e}")
        raise

# === LDA Training ===
# === LDA Training ===
def run_lda_for_app(app_name):
    from gensim.models import LdaModel, CoherenceModel

    # Remove cancel flag for this specific app
    cancel_file = f"cancel_{app_name}.flag"
    if os.path.exists(cancel_file):
        os.remove(cancel_file)

    try:
        print(f"[LDA] Starting LDA training for {app_name}")
        update_progress(0, "Memulai proses training...", app_name)
        
        print(f"[LDA] Loading data for {app_name}")
        df = load_data(app_name)
        if check_cancel(app_name): 
            print(f"[LDA] Training cancelled for {app_name}")
            update_progress(0, "Dibatalkan.", app_name)
            return

        print(f"[LDA] Data loaded successfully for {app_name}, shape: {df.shape}")
        print(f"[LDA] Columns available: {df.columns.tolist()}")

        print(f"[LDA] Preprocessing data for {app_name}")
        update_progress(15, "Preprocessing data...", app_name)
        
        try:
            df, dictionary, corpus = preprocess_data(df)
            print(f"[LDA] Preprocessing completed successfully for {app_name}")
        except Exception as preprocess_error:
            error_msg = f"Preprocessing gagal: {str(preprocess_error)}"
            print(f"[LDA] {error_msg}")
            update_progress(0, f"Training {app_name} gagal: {error_msg}", app_name)
            return  # Exit function early on preprocessing error
        
        if check_cancel(app_name): 
            print(f"[LDA] Training cancelled for {app_name}")
            update_progress(0, "Dibatalkan.", app_name)
            return

        # Verify we have the required column after preprocessing
        if 'bigram_trigram_tokens' not in df.columns:
            error_msg = "kolom 'bigram_trigram_tokens' tidak ditemukan setelah preprocessing"
            print(f"[LDA] Error: {error_msg}")
            update_progress(0, f"Training {app_name} gagal: {error_msg}", app_name)
            return

        texts = df['bigram_trigram_tokens']
        print(f"[LDA] Using {len(texts)} documents for training")

        update_progress(30, "Training LDA dan menghitung coherence...", app_name)

        model_list = []
        coherence_values = []
        total_models = len(range(2, 11))
        
        for i, num_topics in enumerate(range(2, 11)):
            if check_cancel(app_name): 
                print(f"[LDA] Training cancelled for {app_name}")
                update_progress(0, "Dibatalkan.", app_name)
                return
            
            print(f"[LDA] Training model with {num_topics} topics for {app_name}")
            progress_percent = 30 + (i * 60 / total_models)  # 30% to 90%
            update_progress(progress_percent, f"Training model {num_topics} topik ({i+1}/{total_models})...", app_name)
                
            try:
                lda = LdaModel(
                    corpus=corpus,
                    id2word=dictionary,
                    num_topics=num_topics,
                    passes=15,
                    iterations=100,
                    alpha=0.01,
                    eta=0.01,
                    random_state=42
                )
                model_list.append(lda)
                
                print(f"[LDA] Calculating coherence for {num_topics} topics for {app_name}")
                cm = CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence='c_v')
                coherence = cm.get_coherence()

                coherence_values.append(coherence)
                update_progress(progress_percent + 5, f"Model {num_topics} topik: coherence={coherence:.4f}", app_name)
                
            except Exception as model_error:
                print(f"[LDA] Error training model with {num_topics} topics: {model_error}")
                # Continue with next model if one fails
                continue

        if not model_list or not coherence_values:
            error_msg = "Tidak ada model yang berhasil ditraining"
            print(f"[LDA] Error: {error_msg}")
            update_progress(0, f"Training {app_name} gagal: {error_msg}", app_name)
            return

        best_index = coherence_values.index(max(coherence_values))
        best_model = model_list[best_index]
        best_num_topics = range(2, 11)[best_index]
        best_coherence = coherence_values[best_index]

        print(f"[LDA] Best model for {app_name}: {best_num_topics} topics, coherence: {best_coherence:.4f}")
        update_progress(90, f"Menyimpan model terbaik: {best_num_topics} topik (coherence: {best_coherence:.4f})", app_name)

        os.makedirs("models", exist_ok=True)
        with open(f"models/{app_name}_lda.pkl", "wb") as f:
            pickle.dump({
                "model": best_model,
                "corpus": corpus,
                "dictionary": dictionary,
                "coherence": best_coherence
            }, f)

        print(f"[LDA] Training completed successfully for {app_name}")
        update_progress(100, f"Training {app_name} selesai! Model tersimpan dengan {best_num_topics} topik", app_name)
        
    except Exception as e:
        error_msg = f"Error tidak terduga: {str(e)}"
        print(f"[LDA] Training error for {app_name}: {error_msg}")
        import traceback
        traceback.print_exc()
        update_progress(0, f"Training {app_name} gagal: {error_msg}", app_name)
        raise

# === Load Model dan Visualisasi ===
def get_saved_model(app_name):
    try:
        with open(f"models/{app_name}_lda.pkl", "rb") as f:
            data = pickle.load(f)
        
        # Handle both old tuple format and new dict format
        if isinstance(data, dict) and "model" in data:
            lda = data["model"]
            corpus = data["corpus"]
            id2word = data["dictionary"]
        else:
            # Old format: tuple (lda, corpus, id2word)
            lda, corpus, id2word = data
            
        vis = pyLDAvis.gensim_models.prepare(lda, corpus, id2word)
        html = pyLDAvis.prepared_data_to_html(vis)

        # === Responsif: ubah <svg> width & height menjadi fleksibel ===
        html = re.sub(r'(<svg[^>]+)width="[^"]+"', r'\1width="100%"', html)
        html = re.sub(r'(<svg[^>]+)height="[^"]+"', r'\1height="auto"', html)

        # Ubah semua elemen div dengan style width tetap
        html = re.sub(r'style="width:\d+px;"', 'style="width:100%;"', html)

        return html
    except Exception as e:
        return f"<p>Model belum tersedia atau gagal dimuat: {e}</p>"
    
def debug_data_structure(app_name):
    """Debug function to check data structure"""
    path = f"data/data_per_app/{app_name}.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"[DEBUG] File {app_name}.csv exists")
        print(f"[DEBUG] Shape: {df.shape}")
        print(f"[DEBUG] Columns: {df.columns.tolist()}")
        print(f"[DEBUG] First few rows:")
        print(df.head())
        
        # Check for common content columns
        content_columns = []
        for col in df.columns:
            if col.lower() in ['content', 'review', 'text', 'comment', 'body', 'message']:
                content_columns.append(col)
        
        print(f"[DEBUG] Potential content columns: {content_columns}")
        
        if content_columns:
            sample_col = content_columns[0]
            print(f"[DEBUG] Sample data from '{sample_col}':")
            print(df[sample_col].head(3).tolist())
        
        return df
    else:
        print(f"[DEBUG] File {path} does not exist")
        return None