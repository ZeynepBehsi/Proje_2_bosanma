import json
import re
import logging
from pathlib import Path

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

for resource in ['punkt', 'stopwords', 'punkt_tab']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass


class _SimpleTurkishStemmer:
    """Suffix-stripping stemmer for Turkish (SnowballStemmer lacks Turkish support)."""

    _SUFFIXES = [
        'lardaki', 'lerdeki', 'lardan', 'lerden', 'larda', 'lerde',
        'ların', 'lerin', 'larla', 'lerle', 'ları', 'leri', 'lar', 'ler',
        'ndaki', 'ndeki', 'ndan', 'nden', 'nda', 'nde',
        'nun', 'nün', 'nin', 'nın', 'nla', 'nle',
        'daki', 'deki', 'dan', 'den', 'da', 'de',
        'yla', 'yle', 'mak', 'mek', 'acak', 'ecek', 'arak', 'erek',
        'ıyor', 'iyor', 'uyor', 'üyor', 'yor',
        'dı', 'di', 'du', 'dü', 'tı', 'ti', 'tu', 'tü',
        'lık', 'lik', 'luk', 'lük',
        'ması', 'mesi', 'ımız', 'imiz', 'umuz', 'ümüz',
        'ınız', 'iniz', 'unuz', 'ünüz',
        'ım', 'im', 'um', 'üm', 'ın', 'in', 'un', 'ün',
        'yı', 'yi', 'yu', 'yü', 'ya', 'ye',
        'ı', 'i', 'u', 'ü', 'a', 'e',
    ]

    def stem(self, word: str) -> str:
        for suffix in self._SUFFIXES:
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[: -len(suffix)]
        return word


class TurkishTextPreprocessor:
    def __init__(self):
        self.stemmer = _SimpleTurkishStemmer()
        self.stop_words = set(stopwords.words('turkish'))
        self.stop_words.update({
            "ise", "ile", "veya", "ancak", "fakat", "lakin",
            "için", "kadar", "gibi", "daha", "çok", "az",
            "bir", "bu", "şu", "biz", "siz", "onlar", "ne",
            "mi", "mı", "mu", "mü", "var", "yok", "da", "de",
        })
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
        )
        self.logger = logging.getLogger(__name__)

    def tokenize(self, text: str) -> list:
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        tokens = word_tokenize(text, language='turkish')
        return [t for t in tokens if len(t) > 2 and t.isalpha()]

    def remove_stopwords(self, tokens: list) -> list:
        return [t for t in tokens if t not in self.stop_words]

    def stem(self, tokens: list) -> list:
        return [self.stemmer.stem(t) for t in tokens]

    def preprocess(self, text: str, use_stem: bool = True) -> list:
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        if use_stem:
            tokens = self.stem(tokens)
        return tokens

    def preprocess_dataset(self, input_path: str, output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(input_path, encoding='utf-8') as f:
            data = json.load(f)

        toplam = len(data)
        for i, kayit in enumerate(data):
            kayit["soru_tokens"]    = self.preprocess(kayit["soru"])
            kayit["cevap_tokens"]   = self.preprocess(kayit["cevap"])
            kayit["soru_islenmis"]  = " ".join(kayit["soru_tokens"])
            kayit["cevap_islenmis"] = " ".join(kayit["cevap_tokens"])

            if (i + 1) % 50 == 0:
                self.logger.info(f"{i + 1}/{toplam} kayıt işlendi")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.logger.info("Ön işleme tamamlandı.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Türkçe metin ön işleyici")
    parser.add_argument('--input',  default='data/raw/dataset.json')
    parser.add_argument('--output', default='data/processed/dataset_processed.json')
    args = parser.parse_args()

    preprocessor = TurkishTextPreprocessor()
    preprocessor.preprocess_dataset(args.input, args.output)
