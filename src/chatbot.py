import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tfidf_model import TFIDFChatbot
from src.w2v_model import Word2VecChatbot


def main():
    parser = argparse.ArgumentParser(description="Boşanma Hukuku Chatbotu")
    parser.add_argument('--mode', choices=['tfidf', 'w2v', 'compare'], default='tfidf')
    parser.add_argument('--interface', choices=['cli', 'streamlit'], default='cli')
    parser.add_argument('--dataset', default='data/processed/dataset_processed.json')
    parser.add_argument('--tfidf_model', default='data/models/tfidf_model.joblib')
    parser.add_argument('--w2v_model', default='data/models/w2v_model.joblib')
    args = parser.parse_args()

    if args.interface == 'streamlit':
        import subprocess
        subprocess.run(['streamlit', 'run', 'app.py'])
        return

    if args.mode in ('tfidf', 'compare'):
        tfidf_bot = TFIDFChatbot(args.dataset)
        tfidf_bot.load_model(args.tfidf_model)

    if args.mode in ('w2v', 'compare'):
        w2v_bot = Word2VecChatbot(args.dataset)
        w2v_bot.load_model(args.w2v_model)

    if args.mode == 'compare':
        from src.evaluator import ChatbotEvaluator
        evaluator = ChatbotEvaluator()
        test_set = evaluator.load_test_set('data/raw/test_questions.json')
        results = evaluator.compare_models(tfidf_bot, w2v_bot, test_set)
        evaluator.print_report(results)
        return

    bot = tfidf_bot if args.mode == 'tfidf' else w2v_bot
    print(f"\n{'='*55}")
    print(f"  Boşanma Hukuku Bilgi Asistanı  [{args.mode.upper()}]")
    print(f"  Çıkmak için 'çıkış' yazın.")
    print(f"{'='*55}\n")
    bot.interactive_mode()


if __name__ == "__main__":
    main()
