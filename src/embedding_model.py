import json
import logging
from pathlib import Path

import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocess import TurkishTextPreprocessor


class EmbeddingChatbot:
    def __init__(
        self,
        dataset_path: str,
        model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
    ):
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.preprocessor = TurkishTextPreprocessor()
        self.encoder = None
        self.data = []
        self.corpus_vectors = None
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
        )

    def _load_encoder(self) -> None:
        if self.encoder is None:
            self.logger.info(f"Encoder yükleniyor: {self.model_name}")
            self.encoder = SentenceTransformer(self.model_name)
            self.logger.info("Encoder hazır.")

    def load_data(self) -> None:
        with open(self.dataset_path, encoding='utf-8') as f:
            self.data = json.load(f)
        self.logger.info(f"{len(self.data)} kayıt yüklendi")

    def build_index(self) -> None:
        self.load_data()
        self._load_encoder()
        corpus = [kayit["soru"] for kayit in self.data]
        self.logger.info("Corpus encoding başlıyor...")
        self.corpus_vectors = self.encoder.encode(
            corpus,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        self.logger.info(f"Vektörler hazır. Boyut: {self.corpus_vectors.shape}")

    def find_best_match(self, user_query: str, top_k: int = 3) -> list:
        self._load_encoder()
        user_vec = self.encoder.encode([user_query], convert_to_numpy=True)
        scores = cosine_similarity(user_vec, self.corpus_vectors)[0]
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

    def respond(self, user_query: str, threshold: float = 0.30) -> dict:
        matches = self.find_best_match(user_query, top_k=3)
        if not matches or matches[0]["skor"] < threshold:
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
            {
                'model_name': self.model_name,
                'corpus_vectors': self.corpus_vectors,
                'data': self.data,
            },
            path,
        )
        self.logger.info(f"Model kaydedildi: {path}")

    def load_model(self, path: str) -> None:
        obj = joblib.load(path)
        self.model_name = obj['model_name']
        self.corpus_vectors = obj['corpus_vectors']
        self.data = obj['data']
        self.encoder = None  # lazy load için sıfırla
        self.logger.info(f"Model yüklendi: {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
        default='data/processed/dataset_processed.json')
    parser.add_argument('--save',
        default='data/models/embedding_model.joblib')
    args = parser.parse_args()

    bot = EmbeddingChatbot(args.dataset)
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
