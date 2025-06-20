import pickle

with open("models/honkai-star-rail_lda.pkl", "rb") as f:
    data = pickle.load(f)

print(type(data))
if isinstance(data, dict):
    print("✓ Model tersimpan sebagai dictionary")
    print("Key tersedia:", list(data.keys()))
    print("Coherence score:", data.get("coherence"))
else:
    print("❌ Masih berupa tuple")
