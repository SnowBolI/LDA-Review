from flask import Flask, render_template, request, jsonify
from markupsafe import Markup
from lda_utils import run_lda_for_app, get_saved_model, load_data
from chart_storage_functions import save_chart_to_folder,load_chart_from_folder,chart_exists_in_folder,delete_chart_from_folder
import os
import json
import threading
import pandas as pd
from progress_utils import update_progress
import pickle
import pyLDAvis.gensim_models
import math
import pyLDAvis
import numpy as np
from datetime import datetime, timedelta
import traceback

training_sessions = {}
training_threads = {}
MAX_CONCURRENT_TRAINING = 2  # Maksimal 2 training bersamaan
TRAINING_TIMEOUT = 120  # 5 menit timeout jika tidak ada progress
training_lock = threading.Lock()

app = Flask(__name__)  # tidak diubah sesuai permintaan

# Daftar aplikasi
APPS = {
    'honkai-star-rail': 'com.HoYoverse.hkrpgoversea',
    'amazon': 'com.amazon.mShop.android.shopping',
    'wuthering-waves': 'com.kurogame.wutheringwaves.global',
    'arena-breakout': 'com.proximabeta.mf.uamo',
    'spotify': 'com.spotify.music'
}

def cleanup_old_app_data(app_name):
    """Hapus semua data lama untuk aplikasi tertentu"""
    files_to_delete = [
        f"data/data_per_app/{app_name}.csv",  # Data scraping
        f"models/{app_name}_lda.pkl",         # Model LDA
        f"progress_{app_name}.json",          # Progress file
        f"cancel_{app_name}.flag"             # Cancel flag
    ]
    
    deleted_files = []
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                deleted_files.append(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    
    # Hapus chart data
    try:
        delete_chart_from_folder(app_name)
        deleted_files.append(f"Chart data for {app_name}")
        print(f"Deleted chart data for {app_name}")
    except Exception as e:
        print(f"Error deleting chart data for {app_name}: {e}")
    
    return deleted_files


# Initialize progress file
def init_progress_file(app_name=None):
    """Initialize progress file with timestamp for specific app or global"""
    if app_name:
        progress_file = f"progress_{app_name}.json"
    else:
        progress_file = "progress.json"
    
    progress_data = {
        "percent": 0, 
        "description": "Siap untuk training",
        "timestamp": datetime.now().isoformat(),
        "app_name": app_name if app_name else "global"
    }
    with open(progress_file, "w") as f:
        json.dump(progress_data, f)

# Call on startup
if not os.path.exists("progress.json"):
    init_progress_file()

@app.route('/cleanup-data/<app_name>', methods=['POST'])
def cleanup_app_data(app_name):
    """Endpoint untuk hapus semua data aplikasi"""
    try:
        deleted_files = cleanup_old_app_data(app_name)
        return jsonify({
            "status": "success",
            "message": f"Data lama untuk {app_name} berhasil dihapus",
            "deleted_files": deleted_files
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error menghapus data: {str(e)}"
        }), 500


@app.route('/progress')
@app.route('/progress/<app_name>')
def get_progress(app_name=None):
    try:
        if app_name:
            progress_file = f"progress_{app_name}.json"
        else:
            # Jika tidak ada app_name, coba ambil dari query parameter
            app_name = request.args.get('app')
            if app_name:
                progress_file = f"progress_{app_name}.json"
            else:
                progress_file = "progress.json"
        
        if os.path.exists(progress_file):
            with open(progress_file, "r") as f:
                data = json.load(f)
            
            # Add timestamp dan app_name jika belum ada
            if "timestamp" not in data:
                data["timestamp"] = datetime.now().isoformat()
            if "app_name" not in data:
                data["app_name"] = app_name if app_name else "global"
            
            # Debug log
            print(f"[Progress] {app_name}: {data['percent']}% - {data['description']}")
            return jsonify(data)
        else:
            default_data = {
                "percent": 0, 
                "description": "Memulai training...",
                "timestamp": datetime.now().isoformat(),
                "app_name": app_name if app_name else "global"
            }
            print(f"[Progress] {app_name or 'global'}: No progress file, returning default")

            return jsonify(default_data)
    except Exception as e:
        print(f"[Progress] Error reading progress for {app_name}: {e}")
        return jsonify({
            "percent": 0, 
            "description": "Error reading progress",
            "timestamp": datetime.now().isoformat(),
            "app_name": app_name if app_name else "global"
        })
        
    except Exception as e:
        print(f"Error reading progress: {e}")
        return jsonify({
            "percent": 0, 
            "description": "Error reading progress",
            "timestamp": datetime.now().isoformat(),
            "app_name": app_name if app_name else "global"
        })
      
@app.route('/cancel-training', methods=['POST'])
@app.route('/cancel-training/<app_name>', methods=['POST'])
def cancel_training(app_name=None):
    try:
        if app_name:
            cancel_file = f"cancel_{app_name}.flag"
        else:
            # Coba ambil dari request body atau query parameter
            app_name = request.json.get('app') if request.json else request.args.get('app')
            if app_name:
                cancel_file = f"cancel_{app_name}.flag"
            else:
                cancel_file = "cancel.flag"
        
        with open(cancel_file, "w") as f:
            f.write("cancelled")
        return jsonify({"status": "cancelled", "app_name": app_name})
    except Exception as e:
        print(f"Error creating cancel flag: {e}")
        return jsonify({"status": "error", "message": str(e)})
    
@app.route('/')
def index():
    apps = [
        {"slug": "spotify", "name": "Spotify", "developer": "Spotify Ltd."},
        {"slug": "amazon", "name": "Amazon", "developer": "Amazon Mobile LLC"},
        {"slug": "honkai-star-rail", "name": "Honkai: Star Rail", "developer": "Cognosphere PTE. LTD."},
        {"slug": "arena-breakout", "name": "Arena Breakout", "developer": "Level Infinite"},
        {"slug": "wuthering-waves", "name": "Wuthering Waves", "developer": "KURO GAMES"},
    ]
    return render_template("index.html", apps=apps)

def load_scraped_data(app_name):
    path = f"data/data_per_app/{app_name}.csv"
    return pd.read_csv(path) if os.path.exists(path) else None

def load_topic_summary(app_name):
    path = f"models/{app_name}_lda.pkl"
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
        lda_model = data["model"] if isinstance(data, dict) and "model" in data else data[0] if isinstance(data, tuple) else None
        if lda_model is None:
            return None

    topic_words = []
    for i, topic in lda_model.show_topics(formatted=False, num_words=5):
        top_words = ", ".join([word for word, _ in topic])
        topic_words.append({"Topik": i, "Kata Kunci": top_words})
    return pd.DataFrame(topic_words)


def get_client_ip(request):
    """Get client IP address"""
    if request.environ.get('HTTP_X_FORWARDED_FOR'):
        return request.environ['HTTP_X_FORWARDED_FOR'].split(',')[0].strip()
    elif request.environ.get('HTTP_X_REAL_IP'):
        return request.environ['HTTP_X_REAL_IP']
    else:
        return request.environ.get('REMOTE_ADDR', 'unknown')

def cleanup_expired_sessions():
    """Clean up expired training sessions"""
    with training_lock:
        current_time = datetime.now()
        expired_sessions = []
        
        # Cleanup dead threads first
        dead_threads = []
        for app_name, thread in training_threads.items():
            if not thread.is_alive():
                dead_threads.append(app_name)
        
        for app_name in dead_threads:
            del training_threads[app_name]
            print(f"Cleaned up dead thread for app: {app_name}")
        
        # Existing cleanup logic...
        for session_id, session_info in training_sessions.items():
            is_expired = False
            app_name = session_info['app_name']
            progress_file = f"progress_{app_name}.json"
            cancel_file = f"cancel_{app_name}.flag"
            
            try:
                if os.path.exists(cancel_file):
                    is_expired = True
                    print(f"Training cancelled for session: {session_id}")
                elif os.path.exists(progress_file):
                    with open(progress_file, "r") as f:
                        progress_data = json.load(f)
                    
                    progress_percent = progress_data.get("percent", 0)
                    
                    if progress_percent >= 100:
                        is_expired = True
                        print(f"Training completed for session: {session_id}")
                    elif current_time - session_info['start_time'] > timedelta(seconds=TRAINING_TIMEOUT):
                        timestamp_str = progress_data.get("timestamp")
                        if timestamp_str:
                            try:
                                progress_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00').replace('+00:00', ''))
                                if current_time - progress_time > timedelta(seconds=120):
                                    is_expired = True
                                    print(f"Training stuck/timeout for session: {session_id}")
                            except:
                                is_expired = True
                        else:
                            is_expired = True
                            print(f"Training timeout (no timestamp) for session: {session_id}")
                else:
                    if current_time - session_info['start_time'] > timedelta(seconds=60):
                        is_expired = True
                        print(f"No progress file for session: {session_id}")
                        
            except Exception as e:
                print(f"Error checking progress for session {session_id}: {e}")
                is_expired = True
            
            if is_expired:
                expired_sessions.append(session_id)
                try:
                    if os.path.exists(progress_file):
                        with open(progress_file, "r") as f:
                            progress_data = json.load(f)
                        if progress_data.get("percent", 0) < 100:
                            os.remove(progress_file)
                    if os.path.exists(cancel_file):
                        os.remove(cancel_file)
                except Exception as e:
                    print(f"Error cleaning up files for {session_id}: {e}")
        
        for session_id in expired_sessions:
            del training_sessions[session_id]
            print(f"Cleaned up expired training session: {session_id}")
            
def can_start_training(client_ip, app_name):
    """Check if client can start training"""
    cleanup_expired_sessions()
    
    with training_lock:
        session_id = f"{client_ip}_{app_name}"
        
        # Cek apakah ada thread training yang masih aktif untuk app ini
        if app_name in training_threads:
            thread = training_threads[app_name]
            if thread.is_alive():
                # Ada thread aktif, cek progress untuk memberikan info yang akurat
                try:
                    progress_file = f"progress_{app_name}.json"
                    if os.path.exists(progress_file):
                        with open(progress_file, "r") as f:
                            progress_data = json.load(f)
                        progress_percent = progress_data.get("percent", 0)
                        return False, f"Training masih berjalan di background ({progress_percent:.1f}%). Harap tunggu hingga selesai."
                    else:
                        return False, "Training sedang berjalan di background. Harap tunggu."
                except:
                    return False, "Training sedang berjalan di background. Harap tunggu."
            else:
                # Thread sudah mati, bersihkan
                del training_threads[app_name]
        
        # Cek session seperti biasa
        if session_id in training_sessions:
            try:
                progress_file = f"progress_{app_name}.json"
                cancel_file = f"cancel_{app_name}.flag"
                
                if os.path.exists(cancel_file):
                    del training_sessions[session_id]
                    print(f"Removed cancelled session: {session_id}")
                    return True, "OK"
                
                if os.path.exists(progress_file):
                    with open(progress_file, "r") as f:
                        progress_data = json.load(f)
                    
                    progress_percent = progress_data.get("percent", 0)
                    
                    if progress_percent >= 100:
                        del training_sessions[session_id]
                        print(f"Removed completed session: {session_id}")
                        return True, "OK"
                    elif progress_percent > 0:
                        try:
                            timestamp_str = progress_data.get("timestamp")
                            if timestamp_str:
                                progress_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00').replace('+00:00', ''))
                                time_diff = datetime.now() - progress_time
                                
                                if time_diff > timedelta(seconds=120):
                                    del training_sessions[session_id]
                                    print(f"Removed stuck session: {session_id}")
                                    return True, "OK"
                                else:
                                    return False, f"Training sedang berjalan ({progress_percent:.1f}%). Silakan tunggu atau refresh halaman untuk melihat progress."
                            else:
                                return False, f"Training sedang berjalan ({progress_percent:.1f}%). Silakan tunggu atau refresh halaman untuk melihat progress."
                        except Exception as e:
                            print(f"Error parsing timestamp: {e}")
                            return False, f"Training sedang berjalan ({progress_percent:.1f}%). Silakan tunggu atau refresh halaman untuk melihat progress."
                    else:
                        session_start = training_sessions[session_id]['start_time']
                        time_diff = datetime.now() - session_start
                        
                        if time_diff > timedelta(seconds=60):
                            del training_sessions[session_id]
                            print(f"Removed failed session: {session_id}")
                            return True, "OK"
                        else:
                            return False, "Training baru saja dimulai. Silakan tunggu sebentar."
                else:
                    session_start = training_sessions[session_id]['start_time']
                    time_diff = datetime.now() - session_start
                    
                    if time_diff > timedelta(seconds=30):
                        del training_sessions[session_id]
                        print(f"Removed session without progress: {session_id}")
                        return True, "OK"
                    else:
                        return False, "Training sedang diinisialisasi..."
                        
            except Exception as e:
                print(f"Error checking session {session_id}: {e}")
                del training_sessions[session_id]
                return True, "OK"
        
        if len(training_sessions) >= MAX_CONCURRENT_TRAINING:
            active_sessions = []
            for sid, info in training_sessions.items():
                active_sessions.append(f"App: {info['app_name']} (dimulai {info['start_time'].strftime('%H:%M:%S')})")
            
            return False, f"Maksimal {MAX_CONCURRENT_TRAINING} training bersamaan. Sesi aktif: {'; '.join(active_sessions)}"
        
        return True, "OK"
       
def start_training_session(client_ip, app_name):
    """Start a new training session"""
    with training_lock:
        session_id = f"{client_ip}_{app_name}"
        training_sessions[session_id] = {
            'client_ip': client_ip,
            'app_name': app_name,
            'start_time': datetime.now()
        }
        print(f"Started training session: {session_id}")
        return session_id

def end_training_session(client_ip, app_name):
    """End a training session"""
    with training_lock:
        session_id = f"{client_ip}_{app_name}"
        if session_id in training_sessions:
            del training_sessions[session_id]
            print(f"Ended training session: {session_id}")

@app.route('/training-status')
def training_status():
    """Get current training status"""
    cleanup_expired_sessions()
    
    with training_lock:
        active_count = len(training_sessions)
        sessions_info = []
        
        for session_id, info in training_sessions.items():
            sessions_info.append({
                'app_name': info['app_name'],
                'start_time': info['start_time'].strftime('%H:%M:%S'),
                'duration': str(datetime.now() - info['start_time']).split('.')[0]
            })
    
    return jsonify({
        'active_sessions': active_count,
        'max_sessions': MAX_CONCURRENT_TRAINING,
        'can_train': active_count < MAX_CONCURRENT_TRAINING,
        'sessions': sessions_info
    })

def generate_chart_analysis(chart_data, app_name, lda_model):
    """Generate dynamic text analysis for all charts with topic interpretation"""
    
    analysis = {
        'topic_distribution': '',
        'topic_coherence': '',
        'topic_words': '',
        'doc_topic_matrix': '',
        'overall_summary': ''
    }
    
    try:
        # Helper function to interpret topic themes based on keywords
        def interpret_topic_theme(topic_words, topic_id):
            """Interpret what a topic is about based on its keywords"""
            words = [word.lower() for word in topic_words]
            
            # Define theme patterns for different app types
            theme_patterns = {
                # General app themes
                'performance': ['lag', 'slow', 'fast', 'speed', 'performance', 'loading', 'crash', 'bug', 'glitch'],
                'ui_ux': ['interface', 'design', 'ui', 'ux', 'layout', 'screen', 'button', 'menu', 'navigation'],
                'features': ['feature', 'function', 'tool', 'option', 'setting', 'mode', 'update', 'new'],
                'gameplay': ['game', 'play', 'level', 'character', 'battle', 'win', 'lose', 'difficulty'],
                'payment': ['pay', 'price', 'cost', 'money', 'purchase', 'buy', 'expensive', 'cheap', 'free'],
                'social': ['friend', 'chat', 'social', 'share', 'community', 'multiplayer', 'team'],
                'content': ['music', 'song', 'video', 'content', 'quality', 'sound', 'audio'],
                'technical': ['install', 'download', 'version', 'update', 'compatibility', 'system', 'device'],
                'customer_service': ['support', 'help', 'service', 'customer', 'response', 'contact', 'issue'],
                'satisfaction': ['good', 'great', 'excellent', 'amazing', 'love', 'like', 'enjoy', 'satisfied'],
                'dissatisfaction': ['bad', 'terrible', 'hate', 'dislike', 'awful', 'worst', 'disappointed'],
                # E-commerce specific
                'shipping': ['delivery', 'shipping', 'package', 'arrived', 'late', 'fast', 'courier'],
                'product_quality': ['quality', 'original', 'fake', 'genuine', 'damaged', 'broken', 'perfect'],
                'shopping_experience': ['shopping', 'cart', 'checkout', 'order', 'purchase', 'browse'],
                # Gaming specific
                'gacha_monetization': ['gacha', 'summon', 'pull', 'rates', 'pity', 'currency', 'gems', 'crystals'],
                'story_narrative': ['story', 'plot', 'character', 'dialogue', 'quest', 'mission', 'campaign'],
                'graphics_visual': ['graphics', 'visual', 'art', 'animation', 'effect', 'beautiful', 'stunning'],
                # Music streaming specific
                'audio_quality': ['audio', 'sound', 'quality', 'bass', 'treble', 'clear', 'distorted'],
                'playlist_library': ['playlist', 'library', 'collection', 'organize', 'favorite', 'saved'],
                'discovery': ['discover', 'recommendation', 'suggest', 'new', 'artist', 'genre', 'explore']
            }
            
            # Score each theme based on word matches
            theme_scores = {}
            for theme, keywords in theme_patterns.items():
                score = sum(1 for word in words if any(keyword in word or word in keyword for keyword in keywords))
                if score > 0:
                    theme_scores[theme] = score
            
            # Find the best matching theme
            if theme_scores:
                best_theme = max(theme_scores, key=theme_scores.get)
                confidence = theme_scores[best_theme] / len(words)
                
                # Map themes to readable descriptions
                theme_descriptions = {
                    'performance': 'Performa dan Kecepatan Aplikasi',
                    'ui_ux': 'Antarmuka dan Pengalaman Pengguna',
                    'features': 'Fitur dan Fungsionalitas',
                    'gameplay': 'Mekanisme dan Pengalaman Bermain',
                    'payment': 'Sistem Pembayaran dan Harga',
                    'social': 'Fitur Sosial dan Komunitas',
                    'content': 'Kualitas Konten dan Media',
                    'technical': 'Aspek Teknis dan Kompatibilitas',
                    'customer_service': 'Layanan Pelanggan dan Dukungan',
                    'satisfaction': 'Kepuasan dan Pengalaman Positif',
                    'dissatisfaction': 'Ketidakpuasan dan Masalah',
                    'shipping': 'Pengiriman dan Logistik',
                    'product_quality': 'Kualitas Produk',
                    'shopping_experience': 'Pengalaman Berbelanja',
                    'gacha_monetization': 'Sistem Gacha dan Monetisasi',
                    'story_narrative': 'Cerita dan Narasi',
                    'graphics_visual': 'Grafis dan Visual',
                    'audio_quality': 'Kualitas Audio',
                    'playlist_library': 'Playlist dan Koleksi Musik',
                    'discovery': 'Penemuan Konten Baru'
                }
                
                return theme_descriptions.get(best_theme, f"Tema {best_theme.replace('_', ' ').title()}"), confidence
            else:
                return "Tema Umum", 0.0
        
        # 1. Topic Distribution Analysis
        if 'topic_distribution' in chart_data:
            topic_data = chart_data['topic_distribution']['data']
            topic_labels = chart_data['topic_distribution']['labels']
            
            total_weight = sum(topic_data)
            percentages = [(val/total_weight)*100 for val in topic_data]
            
            # Find dominant and minor topics
            max_idx = percentages.index(max(percentages))
            min_idx = percentages.index(min(percentages))
            
            dominant_topic = f"Topik {max_idx + 1}"
            dominant_percent = percentages[max_idx]
            minor_topic = f"Topik {min_idx + 1}"
            minor_percent = percentages[min_idx]
            
            # Calculate topic balance
            std_dev = np.std(percentages)
            balance_status = "seimbang" if std_dev < 10 else "tidak seimbang"
            
            analysis['topic_distribution'] = f"""
            <div class="chart-analysis">
                <h5><i class="fas fa-chart-pie"></i> Analisis Distribusi Topik</h5>
                <p>Dari analisis distribusi topik pada aplikasi <strong>{app_name.replace('-', ' ').title()}</strong>:</p>
                <ul>
                    <li><strong>{dominant_topic}</strong> adalah topik yang paling dominan dengan proporsi <strong>{dominant_percent:.1f}%</strong></li>
                    <li><strong>{minor_topic}</strong> memiliki proporsi terkecil dengan <strong>{minor_percent:.1f}%</strong></li>
                    <li>Distribusi topik relatif <strong>{balance_status}</strong> (standar deviasi: {std_dev:.1f}%)</li>
                    <li>Terdapat <strong>{len(topic_labels)} topik utama</strong> yang teridentifikasi dari review pengguna</li>
                </ul>
                <p><em>Semakin besar proporsi suatu topik, semakin sering tema tersebut muncul dalam review pengguna.</em></p>
            </div>
            """
        
        # 2. Topic Coherence Analysis
        if 'topic_coherence' in chart_data:
            coherence_data = chart_data['topic_coherence']['data']
            coherence_labels = chart_data['topic_coherence']['labels']
            
            avg_coherence = np.mean(coherence_data)
            max_coherence_idx = coherence_data.index(max(coherence_data))
            min_coherence_idx = coherence_data.index(min(coherence_data))
            
            best_topic = f"Topik {max_coherence_idx + 1}"
            worst_topic = f"Topik {min_coherence_idx + 1}"
            
            # Determine overall coherence quality
            if avg_coherence > 0.4:
                quality = "sangat baik"
            elif avg_coherence > 0.3:
                quality = "baik"
            elif avg_coherence > 0.2:
                quality = "cukup"
            else:
                quality = "kurang"
            
            analysis['topic_coherence'] = f"""
            <div class="chart-analysis">
                <h5><i class="fas fa-chart-bar"></i> Analisis Koherensi Topik</h5>
                <p>Evaluasi kualitas topik berdasarkan koherensi kata-kata:</p>
                <ul>
                    <li>Skor koherensi rata-rata: <strong>{avg_coherence:.3f}</strong> (kualitas <strong>{quality}</strong>)</li>
                    <li><strong>{best_topic}</strong> memiliki koherensi tertinggi ({coherence_data[max_coherence_idx]:.3f})</li>
                    <li><strong>{worst_topic}</strong> memiliki koherensi terendah ({coherence_data[min_coherence_idx]:.3f})</li>
                    <li>Rentang koherensi: {min(coherence_data):.3f} - {max(coherence_data):.3f}</li>
                </ul>
                <p><em>Skor koherensi yang tinggi menunjukkan kata-kata dalam topik saling berkaitan dan membentuk tema yang jelas.</em></p>
            </div>
            """
        
        # 3. Enhanced Topic Words Analysis with Theme Interpretation
        if 'topics_words' in chart_data:
            num_topics = len(chart_data['topics_words'])
            
            # Analyze word distribution across topics
            all_words = set()
            topic_word_counts = []
            topic_interpretations = []
            
            for i, topic in enumerate(chart_data['topics_words']):
                words = topic['words']
                all_words.update(words)
                topic_word_counts.append(len(words))
                
                # Interpret topic theme
                theme, confidence = interpret_topic_theme(words, i)
                topic_interpretations.append({
                    'topic_id': i + 1,
                    'theme': theme,
                    'confidence': confidence,
                    'keywords': ', '.join(words[:5])  # Top 5 keywords
                })
            
            avg_words_per_topic = np.mean(topic_word_counts)
            unique_words_total = len(all_words)
            
            # Build topic interpretation table
            interpretation_table = "<div class='topic-interpretation-table'>"
            interpretation_table += "<h6><i class='fas fa-lightbulb'></i> Interpretasi Tema Setiap Topik:</h6>"
            interpretation_table += "<div class='table-responsive'><table class='table table-sm table-striped'>"
            interpretation_table += "<thead><tr><th>Topik</th><th>Tema Utama</th><th>Kata Kunci</th><th>Confidence</th></tr></thead><tbody>"
            
            for interp in topic_interpretations:
                confidence_badge = "success" if interp['confidence'] > 0.3 else "warning" if interp['confidence'] > 0.1 else "secondary"
                interpretation_table += f"""
                <tr>
                    <td><strong>Topik {interp['topic_id']}</strong></td>
                    <td><span class="badge bg-primary">{interp['theme']}</span></td>
                    <td><small>{interp['keywords']}</small></td>
                    <td><span class="badge bg-{confidence_badge}">{interp['confidence']:.2f}</span></td>
                </tr>
                """
            
            interpretation_table += "</tbody></table></div></div>"
            
            analysis['topic_words'] = f"""
            <div class="chart-analysis">
                <h5><i class="fas fa-tags"></i> Analisis Kata-kata Topik</h5>
                <p>Karakteristik kata kunci dalam setiap topik:</p>
                <ul>
                    <li>Total <strong>{num_topics} topik</strong> dengan rata-rata <strong>{avg_words_per_topic:.0f} kata kunci</strong> per topik</li>
                    <li>Terdapat <strong>{unique_words_total} kata unik</strong> yang teridentifikasi sebagai kata kunci</li>
                    <li>Setiap topik menampilkan 10 kata dengan probabilitas tertinggi</li>
                    <li>Gunakan tombol selector untuk melihat distribusi kata pada setiap topik</li>
                </ul>
                
                {interpretation_table}
                
                <p><em>Grafik batang horizontal menunjukkan seberapa kuat asosiasi setiap kata dengan topik yang dipilih. 
                Interpretasi tema didasarkan pada analisis semantik kata-kata kunci dengan confidence score yang menunjukkan 
                tingkat kepercayaan interpretasi.</em></p>
            </div>
            """
        
        # 4. Document-Topic Matrix Analysis
        if 'doc_topic_matrix' in chart_data:
            matrix = np.array(chart_data['doc_topic_matrix'])
            num_docs, num_topics = matrix.shape
            
            # Calculate statistics
            avg_topic_prob = np.mean(matrix, axis=0)
            max_topic_per_doc = np.argmax(matrix, axis=1)
            topic_dominance = np.bincount(max_topic_per_doc, minlength=num_topics)
            
            # Find most common dominant topic
            most_dominant_topic_idx = np.argmax(topic_dominance)
            most_dominant_count = topic_dominance[most_dominant_topic_idx]
            
            # Calculate average maximum probability per document
            max_probs_per_doc = np.max(matrix, axis=1)
            avg_max_prob = np.mean(max_probs_per_doc)
            
            analysis['doc_topic_matrix'] = f"""
            <div class="chart-analysis">
                <h5><i class="fas fa-project-diagram"></i> Analisis Matriks Dokumen-Topik</h5>
                <p>Hubungan antara dokumen (review) dan topik yang teridentifikasi:</p>
                <ul>
                    <li>Menampilkan <strong>{num_docs} dokumen pertama</strong> dari total dataset</li>
                    <li><strong>Topik {most_dominant_topic_idx + 1}</strong> menjadi topik dominan di <strong>{most_dominant_count} dokumen</strong> ({(most_dominant_count/num_docs)*100:.1f}%)</li>
                    <li>Probabilitas topik rata-rata per dokumen: <strong>{avg_max_prob:.3f}</strong></li>
                    <li>Setiap titik menunjukkan kekuatan hubungan antara dokumen dan topik tertentu</li>
                </ul>
                <p><em>Scatter plot membantu memahami bagaimana topik terdistribusi across dokumen dan mengidentifikasi pola dominansi topik.</em></p>
            </div>
            """
        
        # 5. Enhanced Overall Summary with Topic Insights
        if chart_data:
            # Get topic interpretations for summary
            topic_insights = ""
            if 'topics_words' in chart_data and len(chart_data['topics_words']) > 0:
                main_themes = []
                for i, topic in enumerate(chart_data['topics_words']):
                    theme, confidence = interpret_topic_theme(topic['words'], i)
                    if confidence > 0.2:  # Only include high-confidence themes
                        main_themes.append(f"Topik {i+1}: {theme}")
                
                if main_themes:
                    topic_insights = f"""
                    <div class="summary-item">
                        <h6>Tema Utama yang Teridentifikasi</h6>
                        <ul class="theme-list">
                            {"".join(f"<li>{theme}</li>" for theme in main_themes[:3])}
                        </ul>
                    </div>
                    """
            
            analysis['overall_summary'] = f"""
            <div class="chart-analysis overall-summary">
                <h4><i class="fas fa-lightbulb"></i> Ringkasan Analisis Keseluruhan</h4>
                <div class="summary-grid">
                    <div class="summary-item">
                        <h6>Kualitas Model</h6>
                        <p>Model LDA berhasil mengidentifikasi <strong>{len(chart_data.get('topic_distribution', {}).get('labels', []))} topik utama</strong> 
                        dengan tingkat koherensi <strong>{quality}</strong>.</p>
                    </div>
                    <div class="summary-item">
                        <h6>Distribusi Topik</h6>
                        <p>Topik terdistribusi secara <strong>{balance_status}</strong>, menunjukkan 
                        {'keragaman tema yang baik dalam review' if balance_status == 'seimbang' else 'adanya tema dominan tertentu'}.</p>
                    </div>
                    {topic_insights}
                    
                </div>
            </div>
            """
    
    except Exception as e:
        print(f"Error generating chart analysis: {e}")
        traceback.print_exc()
        
        # Fallback analysis
        analysis['overall_summary'] = f"""
        <div class="chart-analysis">
            <h5><i class="fas fa-info-circle"></i> Analisis Visualisasi</h5>
            <p>Visualisasi menampilkan hasil analisis LDA untuk aplikasi <strong>{app_name.replace('-', ' ').title()}</strong>.</p>
            <p>Gunakan grafik interaktif untuk memahami distribusi dan karakteristik topik yang teridentifikasi dari review pengguna.</p>
        </div>
        """
    
    return analysis

def generate_chart_data(lda_model, corpus, dictionary):
    print("Generating chart data...")
    
    # 1. Topic Distribution - PERBAIKAN: Gunakan data yang sama dengan pyLDAvis
    # Menggunakan metode yang sama persis dengan pyLDAvis untuk konsistensi
    try:
        # Siapkan data pyLDAvis terlebih dahulu untuk mendapatkan proporsi yang sama
        vis_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
        vis_dict = vis_data.to_dict()
        
        # Ambil data proporsi topik dari pyLDAvis (lebih akurat)
        mds_data = vis_dict.get("mdsDat", {})
        
        if "Freq" in mds_data:
            # Gunakan data frekuensi dari pyLDAvis
            topic_frequencies = mds_data["Freq"]
            total_freq = sum(topic_frequencies)
            topic_distribution = [float(freq / total_freq) for freq in topic_frequencies]
            
            # Urutkan berdasarkan topic ID jika ada
            if "topics" in mds_data or "Topic" in mds_data:
                topic_col = "topics" if "topics" in mds_data else "Topic"
                topic_ids = mds_data[topic_col]
                
                # Buat mapping dari topic_id ke proporsi
                topic_prop_mapping = {}
                for i, topic_id in enumerate(topic_ids):
                    topic_prop_mapping[int(topic_id) - 1] = topic_distribution[i]  # topic_id dimulai dari 1
                
                # Susun ulang berdasarkan urutan topic ID (0, 1, 2, ...)
                topic_distribution = [topic_prop_mapping.get(i, 0.0) for i in range(lda_model.num_topics)]
            
            print(f"Topic distribution from pyLDAvis: {[f'{x*100:.1f}%' for x in topic_distribution]}")
            
        else:
            raise ValueError("Freq data not found in pyLDAvis")
            
    except Exception as e:
        print(f"Error getting pyLDAvis data, falling back to manual calculation: {e}")
        
        # Fallback: Metode manual yang lebih akurat
        topic_sums = [0.0] * lda_model.num_topics
        total_docs = 0
        
        for doc in corpus:
            doc_topics = lda_model.get_document_topics(doc, minimum_probability=0.0)
            doc_total = sum([prob for _, prob in doc_topics])
            
            if doc_total > 0:  # Hanya hitung dokumen yang valid
                for topic_id, prob in doc_topics:
                    topic_sums[int(topic_id)] += float(prob)
                total_docs += 1
        
        # Normalisasi berdasarkan total dokumen yang valid
        if total_docs > 0:
            topic_distribution = [float(topic_sum / total_docs) for topic_sum in topic_sums]
            # Normalisasi ulang agar total = 1
            total_sum = sum(topic_distribution)
            if total_sum > 0:
                topic_distribution = [float(x / total_sum) for x in topic_distribution]
        else:
            topic_distribution = [1.0 / lda_model.num_topics] * lda_model.num_topics
        
        print(f"Topic distribution (manual): {[f'{x*100:.1f}%' for x in topic_distribution]}")
    
    topic_labels = [f"Topic {i+1}" for i in range(lda_model.num_topics)]
    
    # 2. Topics words data (tidak berubah)
    topics_words_data = []
    for topic_id in range(lda_model.num_topics):
        topic_words = lda_model.show_topic(topic_id, topn=10)
        words = [word for word, prob in topic_words]
        probs = [float(prob) for word, prob in topic_words]
        topics_words_data.append({
            'topic_id': int(topic_id),
            'words': words,
            'probabilities': probs
        })
    
    # 3. Topic coherence (tidak berubah)
    try:
        from gensim.models import CoherenceModel
        
        texts = []
        for doc in corpus:
            doc_words = [dictionary[word_id] for word_id, _ in doc]
            texts.append(doc_words)
        
        topic_coherence_scores = []
        for topic_id in range(lda_model.num_topics):
            topic_words = lda_model.show_topic(topic_id, topn=10)
            topic_word_list = [[word for word, prob in topic_words]]
            
            coherence_model = CoherenceModel(
                topics=topic_word_list,
                texts=texts,
                dictionary=dictionary,
                coherence='c_v'
            )
            coherence_score = coherence_model.get_coherence()
            topic_coherence_scores.append(float(coherence_score))
        
        print(f"Coherence scores (c_v): {[f'{x:.4f}' for x in topic_coherence_scores]}")
        
    except Exception as e:
        print(f"Error calculating coherence: {e}")
        topic_coherence_scores = []
        for topic_id in range(lda_model.num_topics):
            topic_words = lda_model.show_topic(topic_id, topn=5)
            avg_prob = sum([float(prob) for word, prob in topic_words]) / len(topic_words)
            topic_coherence_scores.append(avg_prob)
        
        print(f"Coherence scores (avg prob): {[f'{x:.4f}' for x in topic_coherence_scores]}")
    
    # 4. Document-topic matrix (tidak berubah)
    doc_topic_matrix = []
    for doc_idx, doc in enumerate(corpus[:50]):
        topic_probs = [0.0] * lda_model.num_topics
        doc_topics = lda_model.get_document_topics(doc, minimum_probability=0.0)
        
        for topic_id, prob in doc_topics:
            topic_probs[int(topic_id)] = float(prob)
        
        total_prob = sum(topic_probs)
        if total_prob > 0:
            topic_probs = [p / total_prob for p in topic_probs]
        
        doc_topic_matrix.append(topic_probs)
        
        if doc_idx < 3:
            significant_topics = [(i, p) for i, p in enumerate(topic_probs) if p > 0.01]
            print(f"Doc {doc_idx} topics: {significant_topics}")
    
    # 5. Validasi data
    print("\n=== Data Validation ===")
    print(f"Topic distribution sum: {sum(topic_distribution):.6f}")
    print(f"Topic distribution values: {[f'{x:.6f}' for x in topic_distribution]}")
    print(f"Number of topics: {lda_model.num_topics}")
    print(f"Corpus size: {len(corpus)}")
    print("=== End Validation ===\n")
    
    # 6. Buat chart data
    chart_data = {
        'topic_distribution': {
            'labels': topic_labels,
            'data': topic_distribution  # Sudah dalam bentuk proporsi (0-1), sama dengan pyLDAvis
        },
        'topics_words': topics_words_data,
        'topic_coherence': {
            'labels': topic_labels,
            'data': topic_coherence_scores
        },
        'doc_topic_matrix': doc_topic_matrix
    }
    
    return chart_data
  
@app.route('/lda/<app_name>', methods=['GET', 'POST'])
def lda_page(app_name):
    try:
        model_path = f"models/{app_name}_lda.pkl"

        if request.method == 'POST':
            client_ip = get_client_ip(request)

            # Check if training can be started
            can_train, message = can_start_training(client_ip, app_name)
            if not can_train:
                # Check if it's ongoing training vs error
                if "Training masih berjalan" in message or "Training sedang berjalan" in message:
                    return jsonify({
                        "status": "ongoing",
                        "message": message,
                        "app_name": app_name
                    }), 200  # Return 200 so frontend can handle ongoing training
                else:
                    return jsonify({
                        "status": "error", 
                        "message": message
                    }), 429

            # Start training session
            session_id = start_training_session(client_ip, app_name)
            try:
                # TAMBAHKAN BARIS INI sebelum reset progress:
                # Hapus semua data lama sebelum training
                cleanup_old_app_data(app_name)
                
                # Reset progress file untuk app ini specifically
                progress_file = f"progress_{app_name}.json"
                if os.path.exists(progress_file):
                    os.remove(progress_file)
                init_progress_file(app_name)

                # Remove cancel flag if exists untuk app ini
                cancel_file = f"cancel_{app_name}.flag"
                if os.path.exists(cancel_file):
                    os.remove(cancel_file)

                model_file = f"models/{app_name}_lda.pkl"
                if os.path.exists(model_file):
                    os.remove(model_file)

                # Delete existing chart data when retraining
                delete_chart_from_folder(app_name)

                # Update progress untuk menunjukkan training dimulai
                update_progress(0, f"Training {app_name} dimulai...", app_name)

                # Start training in background thread
                def training_task():
                    try:
                        print(f"[App] Starting training task for {app_name}")
                        
                        # Cek cancel flag sebelum mulai
                        cancel_file = f"cancel_{app_name}.flag"
                        if os.path.exists(cancel_file):
                            print(f"[App] Training cancelled before start for {app_name}")
                            update_progress(0, f"Training {app_name} dibatalkan", app_name)
                            return
                            
                        run_lda_for_app(app_name)
                        
                        # Cek lagi setelah training selesai
                        if os.path.exists(cancel_file):
                            print(f"[App] Training cancelled during execution for {app_name}")
                            update_progress(0, f"Training {app_name} dibatalkan", app_name)
                            return
                        
                        model_file = f"models/{app_name}_lda.pkl"
                        if os.path.exists(model_file):
                            print(f"[App] Training completed successfully for {app_name}")
                            update_progress(100, f"Training {app_name} selesai!", app_name)
                            return
                            
                    except Exception as e:
                        error_msg = str(e)
                        print(f"[App] Training error for {app_name}: {error_msg}")
                        import traceback
                        traceback.print_exc()
                        update_progress(0, f"Training {app_name} gagal: {error_msg}", app_name)
                    finally:
                        # Cleanup thread tracking dan session
                        print(f"[App] Ending training session for {app_name}")
                        end_training_session(client_ip, app_name)
                        
                        # Remove from thread tracking
                        with training_lock:
                            if app_name in training_threads:
                                del training_threads[app_name]
                                print(f"[App] Removed thread tracking for {app_name}")

                # Start training thread dengan tracking
                training_thread = threading.Thread(target=training_task)
                training_thread.daemon = True

                # Track thread sebelum start
                with training_lock:
                    training_threads[app_name] = training_thread

                training_thread.start()           

                # Start training thread
                training_thread = threading.Thread(target=training_task)
                training_thread.daemon = True
                training_thread.start()

                return jsonify({"status": "started", "app_name": app_name})

            except Exception as e:
                # End training session on error
                end_training_session(client_ip, app_name)
                return jsonify({
                    "status": "error",
                    "message": f"Training gagal dimulai: {str(e)}"
                }), 500

        # GET request handling - load and display data
        df = load_scraped_data(app_name)
        head_html = df.head().to_html(classes="table table-striped", index=False) if df is not None else "<p>Data scraping tidak ditemukan.</p>"
        tail_html = df.tail().to_html(classes="table table-striped", index=False) if df is not None else "<p>Data scraping tidak ditemukan.</p>"

        topic_summary_df = load_topic_summary(app_name)
        topic_summary_html = topic_summary_df.to_html(classes="table table-striped", index=False) if topic_summary_df is not None else "<p><i class='fas fa-exclamation-triangle'></i> Model belum tersedia.</p>"

        model_data = coherence_score = model_available = vis = topic_descriptions = pyldavis_explanation = chart_data = chart_analysis = None

        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                loaded_data = pickle.load(f)

            if isinstance(loaded_data, dict) and 'model' in loaded_data:
                model_data = loaded_data
                coherence_score = loaded_data.get("coherence")
                model_available = True

                lda_model = loaded_data.get("model")
                corpus = loaded_data.get("corpus")
                dictionary = loaded_data.get("dictionary")

                # Check if chart data already exists
                if chart_exists_in_folder(app_name):
                    print(f"Loading existing chart data for {app_name}")
                    chart_data, chart_analysis = load_chart_from_folder(app_name)
                else:
                    print(f"Generating new chart data for {app_name}")
                    chart_data = generate_chart_data(lda_model, corpus, dictionary)
                    chart_analysis = generate_chart_analysis(chart_data, app_name, lda_model)
                    save_chart_to_folder(app_name, chart_data, chart_analysis)

                vis_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
                vis = pyLDAvis.prepared_data_to_html(vis_data)

                topics_df = pd.DataFrame(vis_data.to_dict()["mdsDat"])
                topic_col = "topics" if "topics" in topics_df.columns else "Topic"
                freq_col = "Freq" if "Freq" in topics_df.columns else "freq"
                total_freq = topics_df[freq_col].sum()

                topic_descriptions = f"<p>Model ini menghasilkan <strong>{lda_model.num_topics}</strong> topik.</p><div class='table-responsive'><table class='table table-bordered table-striped table-hover'><thead class='table-primary'><tr><th>Topik</th><th>Proporsi (%)</th><th>Kata Kunci</th></tr></thead><tbody>"
                pyldavis_topic_explanations = []

                for _, row in topics_df.iterrows():
                    try:
                        topic_id = int(row[topic_col]) - 1
                        size = (float(row[freq_col]) / total_freq) * 100
                        top_terms = lda_model.show_topic(topic_id, topn=5)

                        if isinstance(top_terms, list) and all(isinstance(t, tuple) for t in top_terms):
                            keywords = ", ".join([term for term, _ in top_terms])
                            keywords_html = "<ul>" + "".join(f"<li>{term}: {weight:.3f}</li>" for term, weight in top_terms) + "</ul>"
                        else:
                            raise ValueError(f"top_terms format salah: {top_terms}")

                        topic_descriptions += f"<tr><td><strong>Topik {topic_id+1}</strong></td><td>{size:.2f}</td><td><em>{keywords}</em></td></tr>"
                        x, y = row.get("x", 0), row.get("y", 0)
                        pyldavis_topic_explanations.append(f"<li><strong>Topik {topic_id+1} (x={x:.2f}, y={y:.2f})</strong>: {keywords_html}</li>")
                    except Exception as e:
                        print(f"Error saat memproses row: {row}")
                        traceback.print_exc()

                topic_descriptions += "</tbody></table></div>"
                pyldavis_explanation = f"""
                <h5>Penjelasan Visualisasi PyLDAvis</h5>
                <ul>
                    <li><strong>Bubble Chart:</strong> Tiap lingkaran mewakili topik. Ukuran = proporsi.</li>
                    <li><strong>Warna & Posisi:</strong> Mewakili kedekatan antar topik (reduksi dimensi).</li>
                    <li><strong>Kata Kunci Panel Kanan:</strong> Menunjukkan bobot tiap kata dalam topik.</li>
                    {''.join(pyldavis_topic_explanations)}
                    <li><strong>Slider Relevansi:</strong> Mengatur kata berdasarkan spesifisitas ke topik.</li>
                </ul>
                """

        return render_template(
            "lda.html",
            app_name=app_name,
            head_html=Markup(head_html),
            tail_html=Markup(tail_html),
            topic_summary=Markup(topic_summary_html),
            model_data=model_data,
            model_available=model_available,
            coherence_score=coherence_score,
            vis=Markup(vis),
            topic_descriptions=Markup(topic_descriptions),
            pyldavis_explanation=Markup(pyldavis_explanation),
            chart_data=chart_data,
            chart_analysis=chart_analysis
        )
    except Exception as e:
        traceback.print_exc()
        return f"<h2>Terjadi Error:</h2><pre>{str(e)}</pre>", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)