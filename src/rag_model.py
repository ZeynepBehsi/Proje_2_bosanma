import os
import json
import logging
from dotenv import load_dotenv
from groq import Groq
from src.embedding_model import EmbeddingChatbot


class RAGChatbot:
    def __init__(
        self,
        dataset_path: str,
        embedding_model_path: str,
        groq_model: str = "llama-3.3-70b-versatile",
    ):
        load_dotenv()
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.groq_model = groq_model
        self.retriever = EmbeddingChatbot(dataset_path)
        self.retriever.load_model(embedding_model_path)
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
        )

    def _build_prompt(self, user_query: str, contexts: list) -> tuple:
        system = """Sen Türk hukuku konusunda bilgi veren bir asistansın.
Görevin: Sana verilen hukuki bilgilere dayanarak kullanıcının
sorusunu Türkçe, anlaşılır ve dostane bir şekilde yanıtlamak.

ÖNEMLİ KURALLAR:
- Yalnızca aşağıdaki bilgilere dayan, dışına çıkma
- Madde numaralarını ve kaynakları mutlaka belirt
- Eğer bilgiler soruyu tam karşılamıyorsa bunu açıkça söyle
- Cevabın sonuna 'Bu bilgiler genel amaçlıdır, bir avukattan destek almanızı öneririz.' ekle
- Asla uydurma bilgi verme"""

        context_text = "\n---\n".join(
            f"[Kaynak: {c['kaynak']} | Kategori: {c['kategori']}]\n"
            f"Soru: {c['soru']}\nBilgi: {c['cevap']}"
            for c in contexts
        )

        return system, context_text

    def respond(self, user_query: str, top_k: int = 3) -> dict:
        matches = self.retriever.find_best_match(user_query, top_k=top_k)

        if not matches or matches[0]["skor"] < 0.30:
            return {
                "cevap": "Bu konuda yeterli bilgi bulunamadı. "
                         "Lütfen bir hukuk uzmanına danışın.",
                "kaynaklar": [],
                "skor": 0.0,
                "rag": False,
            }

        system, context_text = self._build_prompt(user_query, matches)

        completion = self.groq_client.chat.completions.create(
            model=self.groq_model,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": (
                        f"Aşağıdaki hukuki bilgileri kullanarak soruyu yanıtla:\n\n"
                        f"{context_text}\n\n"
                        f"Kullanıcı sorusu: {user_query}"
                    ),
                },
            ],
            temperature=0.3,
            max_tokens=1024,
        )

        generated = completion.choices[0].message.content

        return {
            "cevap": generated,
            "kaynaklar": [
                {"kaynak": m["kaynak"], "kategori": m["kategori"], "skor": m["skor"]}
                for m in matches
            ],
            "skor": matches[0]["skor"],
            "rag": True,
        }

    def print_response(self, user_query: str) -> None:
        result = self.respond(user_query)
        print("\n" + "=" * 60)
        print(f"SORU: {user_query}")
        print("-" * 60)
        print(f"CEVAP:\n{result['cevap']}")
        if result["kaynaklar"]:
            print("-" * 60)
            print("KAYNAKLAR:")
            for k in result["kaynaklar"]:
                print(f"  • {k['kaynak']} ({k['kategori']}) — skor: {k['skor']:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='data/processed/dataset_processed.json')
    parser.add_argument('--embedding_model', default='data/models/embedding_model.joblib')
    args = parser.parse_args()

    bot = RAGChatbot(args.dataset, args.embedding_model)

    test_sorular = [
        "Eşimden boşanmak istiyorum, nereden başlamalıyım?",
        "Çocuğumun velayetini nasıl alabilirim?",
        "Şiddet görüyorum, ne yapabilirim?",
    ]
    for soru in test_sorular:
        bot.print_response(soru)
