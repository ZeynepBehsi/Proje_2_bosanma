import json
import time
import logging
from pathlib import Path

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from src.tfidf_model import TFIDFChatbot
from src.w2v_model import Word2VecChatbot
from src.embedding_model import EmbeddingChatbot


class ChatbotEvaluator:
    def __init__(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
        )
        self.logger = logging.getLogger(__name__)

    def load_test_set(self, path: str) -> list:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            item["beklenen_cevap_id"] = int(item["beklenen_cevap_id"])
        self.logger.info(f"{len(data)} test sorusu yüklendi")
        return data

    def evaluate(self, chatbot, test_set: list, model_name: str) -> dict:
        top1_dogru = 0
        top3_dogru = 0
        kat_dogru = 0
        yanlis_eslesme = []
        sure_listesi = []

        for test_item in test_set:
            t0 = time.time()
            matches = chatbot.find_best_match(test_item["soru"], top_k=3)
            sure = time.time() - t0
            sure_listesi.append(sure)

            if not matches:
                continue

            matched_id = next(
                (d["id"] for d in chatbot.data if d["soru"] == matches[0]["soru"]),
                None,
            )

            if matched_id == test_item["beklenen_cevap_id"]:
                top1_dogru += 1

            if matches[0]["kategori"] == test_item["beklenen_kategori"]:
                kat_dogru += 1

            top3_ids = [
                next(
                    (d["id"] for d in chatbot.data if d["soru"] == m["soru"]),
                    None,
                )
                for m in matches
            ]
            if test_item["beklenen_cevap_id"] in top3_ids:
                top3_dogru += 1
            else:
                yanlis_eslesme.append({
                    "soru": test_item["soru"],
                    "beklenen_id": test_item["beklenen_cevap_id"],
                    "beklenen_kategori": test_item["beklenen_kategori"],
                    "bulunan_soru": matches[0]["soru"] if matches else "",
                    "bulunan_kategori": matches[0]["kategori"] if matches else "",
                    "skor": matches[0]["skor"] if matches else 0.0,
                })

        n = len(test_set)
        return {
            "model": model_name,
            "test_sayisi": n,
            "top1_accuracy": round(top1_dogru / n, 4),
            "top3_accuracy": round(top3_dogru / n, 4),
            "kategori_accuracy": round(kat_dogru / n, 4),
            "ort_sure_ms": round(np.mean(sure_listesi) * 1000, 2),
            "yanlis_eslesme": yanlis_eslesme,
        }

    def compare_models(self, tfidf_chatbot, w2v_chatbot, test_set: list) -> dict:
        self.logger.info("TF-IDF modeli değerlendiriliyor...")
        tfidf_result = self.evaluate(tfidf_chatbot, test_set, "TF-IDF")

        self.logger.info("Word2Vec modeli değerlendiriliyor...")
        w2v_result = self.evaluate(w2v_chatbot, test_set, "Word2Vec")

        Path("data/results").mkdir(parents=True, exist_ok=True)

        with open("data/results/evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump([tfidf_result, w2v_result], f, ensure_ascii=False, indent=2)

        if HAS_PANDAS:
            df = pd.DataFrame([
                {k: v for k, v in r.items() if k != "yanlis_eslesme"}
                for r in [tfidf_result, w2v_result]
            ])
            df.to_csv("data/results/comparison.csv", index=False)

        return {"tfidf": tfidf_result, "w2v": w2v_result}

    def compare_all_models(self, tfidf_chatbot, w2v_chatbot,
                           embedding_chatbot, test_set: list) -> dict:
        self.logger.info("TF-IDF modeli değerlendiriliyor...")
        tfidf_result = self.evaluate(tfidf_chatbot, test_set, "TF-IDF")

        self.logger.info("Word2Vec modeli değerlendiriliyor...")
        w2v_result = self.evaluate(w2v_chatbot, test_set, "Word2Vec")

        self.logger.info("Embedding modeli değerlendiriliyor...")
        emb_result = self.evaluate(embedding_chatbot, test_set, "Embedding")

        Path("data/results").mkdir(parents=True, exist_ok=True)

        with open("data/results/evaluation_all_models.json", "w", encoding="utf-8") as f:
            json.dump([tfidf_result, w2v_result, emb_result],
                      f, ensure_ascii=False, indent=2)

        if HAS_PANDAS:
            df = pd.DataFrame([
                {k: v for k, v in r.items() if k != "yanlis_eslesme"}
                for r in [tfidf_result, w2v_result, emb_result]
            ])
            df.to_csv("data/results/comparison_all.csv", index=False)
            self.logger.info("Karşılaştırma CSV kaydedildi.")

        return {
            "tfidf": tfidf_result,
            "w2v": w2v_result,
            "embedding": emb_result,
        }

    def print_report(self, results: dict) -> None:
        print("\n" + "=" * 60)
        print("MODEL KARŞILAŞTIRMA RAPORU")
        print("=" * 60)
        for model_adi, r in results.items():
            print(f"\n[{r['model']}]")
            print(f"  Top-1 Accuracy : {r['top1_accuracy']:.2%}")
            print(f"  Top-3 Accuracy : {r['top3_accuracy']:.2%}")
            print(f"  Kategori Acc.  : {r['kategori_accuracy']:.2%}")
            print(f"  Ort. Süre (ms) : {r['ort_sure_ms']}")
            print(f"  Yanlış Eşleşme: {len(r['yanlis_eslesme'])}/{r['test_sayisi']}")
        print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--test',
        default='data/raw/test_questions.json')
    parser.add_argument('--tfidf_model',
        default='data/models/tfidf_model.joblib')
    parser.add_argument('--w2v_model',
        default='data/models/w2v_model.joblib')
    parser.add_argument('--embedding_model',
        default='data/models/embedding_model.joblib')
    args = parser.parse_args()

    tfidf_bot = TFIDFChatbot('data/processed/dataset_processed.json')
    tfidf_bot.load_model(args.tfidf_model)

    w2v_bot = Word2VecChatbot('data/processed/dataset_processed.json')
    w2v_bot.load_model(args.w2v_model)

    emb_bot = EmbeddingChatbot('data/processed/dataset_processed.json')
    emb_bot.load_model(args.embedding_model)

    evaluator = ChatbotEvaluator()
    test_set  = evaluator.load_test_set(args.test)
    results   = evaluator.compare_all_models(
                    tfidf_bot, w2v_bot, emb_bot, test_set)
    evaluator.print_report(results)
