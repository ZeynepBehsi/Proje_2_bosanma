# ⚖️ Boşanma Davaları Hukuki Bilgi Chatbotu

> Türk Medeni Kanunu kapsamında NLP tabanlı hukuki bilgilendirme sistemi.  
> Bu sistem yalnızca genel bilgi amaçlıdır, hukuki danışmanlık yerine geçmez.

## 🎯 Proje Hakkında

Bu proje, Türk boşanma hukuku alanında kullanıcıların sık sorduğu sorulara
TMK'ya dayalı yanıtlar sunan bir chatbot sistemidir. Dört farklı NLP yöntemi
karşılaştırmalı olarak uygulanmış ve değerlendirilmiştir:

- **TF-IDF + Cosine Similarity** — anahtar kelime tabanlı
- **Word2Vec + Cosine Similarity** — anlam tabanlı
- **Sentence Transformer** — çok dilli embedding (en iyi doğruluk)
- **RAG + Groq LLaMA 3.3 70B** — doğal Türkçe üretim

## 📊 Model Performansı (50 test sorusu)

| Model | Kategori Accuracy | Top-3 Accuracy | Ort. Süre |
|-------|:-----------------:|:--------------:|:---------:|
| TF-IDF | %60 | %10 | 1.59 ms |
| Word2Vec | %58 | %10 | 0.29 ms |
| **Embedding** | **%76** | **%16** | 178 ms |
| RAG + Groq | nitel değerlendirme | — | ~1-2 sn |

## 🗂️ Proje Yapısı

```
Proje_2_bosanma_davalari/
├── data/
│   ├── raw/                    # Ham veri (250 S&C çifti, 50 test sorusu)
│   ├── processed/              # NLP ön işlenmiş veri
│   ├── models/                 # Eğitilmiş model dosyaları (.joblib)
│   └── results/                # Değerlendirme sonuçları (CSV, JSON)
├── src/
│   ├── preprocess.py           # Türkçe NLP pipeline (tokenize, stopword, stem)
│   ├── tfidf_model.py          # Model 1: TF-IDF
│   ├── w2v_model.py            # Model 2: Word2Vec (Gensim)
│   ├── embedding_model.py      # Model 3: Sentence Transformer
│   ├── rag_model.py            # Model 4: RAG + Groq LLM
│   ├── evaluator.py            # Karşılaştırmalı değerlendirme
│   └── chatbot.py              # CLI giriş noktası
├── notebooks/
│   └── exploration.ipynb       # Keşifsel veri analizi
├── tests/
│   └── test_models.py          # Birim testler
├── app.py                      # Streamlit web arayüzü
├── requirements.txt
└── .env                        # API anahtarları (git'e eklenmez)
```

## ⚙️ Kurulum

```bash
# 1. Repoyu klonla
git clone <repo-url>
cd Proje_2_bosanma_davalari

# 2. Sanal ortam oluştur
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows

# 3. Bağımlılıkları yükle
pip install -r requirements.txt

# 4. API anahtarını ayarla (.env dosyası oluştur)
echo "GROQ_API_KEY=your_key_here" > .env
```

## 🚀 Kullanım

```bash
# Streamlit arayüzü (önerilen)
PYTHONPATH=. python -m streamlit run app.py

# Model karşılaştırma raporu
PYTHONPATH=. python src/evaluator.py

# CLI modu
PYTHONPATH=. python src/chatbot.py --mode tfidf --interface cli
PYTHONPATH=. python src/chatbot.py --mode compare --interface cli

# Modelleri sıfırdan eğitmek için
PYTHONPATH=. python src/preprocess.py
PYTHONPATH=. python src/tfidf_model.py
PYTHONPATH=. python src/w2v_model.py
PYTHONPATH=. python src/embedding_model.py
```

## 🧠 Kullanılan Teknolojiler

| Teknoloji | Amaç |
|-----------|------|
| scikit-learn | TF-IDF vektörizasyon, cosine similarity |
| Gensim | Word2Vec eğitimi |
| sentence-transformers | Çok dilli embedding (paraphrase-multilingual-MiniLM-L12-v2) |
| Groq API | LLaMA 3.3 70B ile RAG üretimi |
| NLTK | Türkçe tokenizasyon, stopword |
| Streamlit | Web arayüzü |
| joblib | Model kaydetme/yükleme |
| python-dotenv | API key yönetimi |

## 📁 Veri Seti

- **250 soru-cevap çifti**, 7 kategori
- Kategoriler: boşanma_süreci, nafaka, velayet, mal_paylaşımı,
  tazminat, şiddet_ve_tedbir, yurt_dışı_boşanma
- Kaynaklar: TMK Madde 161-186, 321-324, 6284 Sayılı Kanun
- **50 test sorusu** — veri setindekilerden farklı ifadelerle

## ⚠️ Yasal Uyarı

Bu sistem yalnızca genel bilgilendirme amaçlıdır.  
Hukuki danışmanlık yerine geçmez.  
Hukuki konularda mutlaka bir avukattan destek alınız.
