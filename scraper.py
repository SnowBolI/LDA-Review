import os
from google_play_scraper import reviews, Sort
import pandas as pd

def scrape_reviews(app_name, lang='id', country='id', total=20000):
    app_ids = {
        'honkai-star-rail': 'com.HoYoverse.hkrpgoversea',
        'amazon': 'com.amazon.mShop.android.shopping',
        'wuthering-waves': 'com.kurogame.wutheringwaves.global',
        'arena-breakout': 'com.proximabeta.mf.uamo',
        'spotify': 'com.spotify.music'
    }

    app_id = app_ids.get(app_name)
    if not app_id:
        raise ValueError(f"Aplikasi '{app_name}' tidak ditemukan.")

    # TAMBAHAN BARU: Hapus data lama sebelum scraping
    data_file = f"data/data_per_app/{app_name}.csv"
    if os.path.exists(data_file):
        try:
            os.remove(data_file)
            print(f"Data lama {app_name} telah dihapus: {data_file}")
        except Exception as e:
            print(f"Warning: Gagal menghapus data lama {data_file}: {e}")

    # Pastikan direktori ada
    os.makedirs("data/data_per_app", exist_ok=True)

    all_reviews = []
    count = 0
    next_token = None

    while count < total:
        try:
            rvs, next_token = reviews(
                app_id,
                lang='en',
                country='us',
                sort=Sort.NEWEST,
                count=200,  # maksimal per batch
                continuation_token=next_token
            )
            if not rvs:
                break
            all_reviews.extend(rvs)
            count += len(rvs)
            print(f"Progress: {count}/{total}")
        except Exception as e:
            print(f"Error: {e}")
            break

    df = pd.DataFrame(all_reviews)
    
    # TAMBAHAN BARU: Simpan data baru
    final_df = df[['content', 'score', 'at']]  # kolom yang relevan
    final_df.to_csv(data_file, index=False)
    print(f"Data baru disimpan: {data_file}")
    
    head_html = df.head().to_html(classes="table table-striped", index=False)
    tail_html = df.tail().to_html(classes="table table-striped", index=False)
    return final_df