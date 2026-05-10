import json
import logging
from pathlib import Path

import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocess import TurkishTextPreprocessor


class TFIDFChatbot:
    def __init__(self, dataset_path: str, ngram_range=(1, 2), max_features=5000):
        self.dataset_path = dataset_path
        self.preprocessor = TurkishTextPreprocessor()
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            analyzer='word',
        )
        self.vectors = None
        self.data = []
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self) -> None:
        with open(self.dataset_path, encoding='utf-8') as f:
            self.data = json.load(f)
        self.logger.info(f"{len(self.data)} kayıt yüklendi")

    def build_index(self) -> None:
        self.load_data()
        corpus = [
            kayit.get(
                "soru_islenmis",
                " ".join(self.preprocessor.preprocess(kayit["soru"])),
            )
            for kayit in self.data
        ]
        self.vectors = self.vectorizer.fit_transform(corpus)
        self.logger.info(f"İndeks oluşturuldu. Matris boyutu: {self.vectors.shape}")

    def find_best_match(self, user_query: str, top_k: int = 3) -> list:
        user_tokens = self.preprocessor.preprocess(user_query)
        user_text = " ".join(user_tokens)
        user_vec = self.vectorizer.transform([user_text])
        scores = cosine_similarity(user_vec, self.vectors)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "soru": self.data[i]["soru"],
                "cevap": self.data[i]["cevap"],
                "kategori": self.data[i]["kategori"],
                "kaynak": self.data[i]["kaynak"],
                "skor": float(scores[i]),
            }
            for i in top_indices
        ]

    def respond(self, user_query: str, threshold: float = 0.15) -> dict:
        matches = self.find_best_match(user_query, top_k=3)
        if matches[0]["skor"] < threshold:
            return {
                "cevap": (
                    "Üzgünüm, bu konuda bilgim bulunmuyor. "
                    "Lütfen bir hukuk uzmanına danışın."
                ),
                "kategori": "bilinmiyor",
                "kaynak": "-",
                "skor": 0.0,
                "alternatifler": [],
            }
        return {
            "cevap": matches[0]["cevap"],
            "kategori": matches[0]["kategori"],
            "kaynak": matches[0]["kaynak"],
            "skor": matches[0]["skor"],
            "alternatifler": matches[1:],
        }

    def print_response(self, user_query: str) -> None:
        result = self.respond(user_query)
        print("=" * 60)
        print(f"SORU    : {user_query}")
        print(f"KATEGORİ: {result['kategori']}")
        print(f"KAYNAK  : {result['kaynak']}")
        print(f"SKOR    : {result['skor']:.4f}")
        print(f"CEVAP   :\n{result['cevap']}")
        print("=" * 60)

    def interactive_mode(self) -> None:
        print("Çıkmak için 'çıkış' veya 'exit' yazın.")
        while True:
            user_query = input("Sorunuz: ").strip()
            if user_query.lower() in ("çıkış", "exit"):
                break
            self.print_response(user_query)

    def save_model(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"vectorizer": self.vectorizer, "vectors": self.vectors, "data": self.data},
            path,
        )
        self.logger.info(f"Model kaydedildi: {path}")

    def load_model(self, path: str) -> None:
        obj = joblib.load(path)
        self.vectorizer = obj["vectorizer"]
        self.vectors = obj["vectors"]
        self.data = obj["data"]
        self.logger.info(f"Model yüklendi: {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/dataset_processed.json")
    parser.add_argument("--save", default="data/models/tfidf_model.joblib")
    args = parser.parse_args()

    bot = TFIDFChatbot(args.dataset)
    bot.build_index()
    bot.save_model(args.save)

    test_sorular = [
        "Boşanma davası nasıl açılır?",
        "Nafaka ne kadar sürer?",
        "Çocuğun velayeti kime verilir?",
        "Mal paylaşımında hangi kurallar geçerli?",
        "Şiddet durumunda ne yapabilirim?",
    ]
    for soru in test_sorular:
        bot.print_response(soru)
