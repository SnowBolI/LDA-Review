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
    head_html = df.head().to_html(classes="table table-striped", index=False)
    tail_html = df.tail().to_html(classes="table table-striped", index=False)
    return df[['content', 'score', 'at']]  # kolom yang relevan
