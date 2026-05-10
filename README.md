# Boşanma Davaları Hukuki Bilgi Chatbotu

## Proje Hakkında

Bu proje, Türk Medeni Kanunu kapsamındaki boşanma davası metinlerinden derlenen 250 kayıtlık bir veri setiyle NLP tabanlı bir hukuki bilgi chatbotu geliştirmeyi amaçlamaktadır. TF-IDF, Word2Vec ve çok dilli Sentence Transformer olmak üç farklı metin temsil yöntemi uygulanmış ve karşılaştırılmıştır. Proje hem komut satırı hem de Streamlit web arayüzü üzerinden kullanılabilir.

## Kullanılan Teknolojiler

- Python 3.10
- scikit-learn (TF-IDF)
- Gensim (Word2Vec)
- Sentence Transformers (paraphrase-multilingual-MiniLM-L12-v2)
- Streamlit
- NLTK

## Model Performansı

| Model | Kategori Accuracy | Hız |
|-------|------------------|-----|
| TF-IDF | %60 | 1.59ms |
| Word2Vec | %58 | 0.29ms |
| Embedding | %76 | 178ms |

## Kurulum

```bash
git clone <repo-url>
cd Proje_2_bosanma_davalari
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Kullanım

```bash
# Streamlit arayüzü
PYTHONPATH=. .venv/bin/python -m streamlit run app.py

# Model karşılaştırma raporu
PYTHONPATH=. .venv/bin/python src/evaluator.py
```

## Proje Yapısı

```
Proje_2_bosanma/
├── data/
│   ├── raw/               # Ham veri ve test soruları
│   ├── processed/         # İşlenmiş veri seti (250 kayıt)
│   ├── models/            # Eğitilmiş model dosyaları (.joblib)
│   └── results/           # Değerlendirme sonuçları
├── src/
│   ├── __init__.py
│   ├── preprocess.py      # Türkçe metin ön işleme (NLTK)
│   ├── tfidf_model.py     # TF-IDF + cosine similarity
│   ├── w2v_model.py       # Word2Vec (Gensim) + cosine similarity
│   ├── embedding_model.py # Sentence Transformer (çok dilli)
│   ├── evaluator.py       # 3 model karşılaştırma ve raporlama
│   └── chatbot.py         # CLI arayüzü
├── notebooks/
│   └── exploration.ipynb  # Keşifsel veri analizi
├── tests/
│   └── test_models.py     # Birim testler
├── app.py                 # Streamlit web arayüzü
├── requirements.txt
└── README.md
```

## Uyarı

Bu sistem yalnızca genel bilgi amaçlıdır, hukuki danışmanlık yerine geçmez.
