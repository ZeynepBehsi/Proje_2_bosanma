import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from src.tfidf_model import TFIDFChatbot
from src.w2v_model import Word2VecChatbot
from src.embedding_model import EmbeddingChatbot

st.set_page_config(
    page_title="Boşanma Hukuku Asistanı",
    page_icon="⚖️",
    layout="centered",
)

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Ayarlar")
    model_sec = st.radio("Model", ["TF-IDF", "Word2Vec", "Embedding (en iyi)"])
    threshold = st.slider("Eşik skoru", 0.05, 0.50, 0.15, 0.05)
    if model_sec == "TF-IDF":
        st.info("⚡ Hızlı — Anahtar kelime eşleşmesi\n\n"
                "Kategori doğruluğu: %60")
    elif model_sec == "Word2Vec":
        st.info("🧠 Orta — Anlam tabanlı eşleşme\n\n"
                "Kategori doğruluğu: %58")
    else:
        st.success("🎯 En iyi — Çok dilli embedding\n\n"
                   "Kategori doğruluğu: %76")
    if st.button("🗑️ Sohbeti temizle"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.caption(
        "Bu uygulama yalnızca genel bilgi amaçlıdır. "
        "Hukuki danışmanlık yerine geçmez."
    )

# ── Model yükleme (cache) ─────────────────────────────────────────────
@st.cache_resource
def load_tfidf():
    bot = TFIDFChatbot('data/processed/dataset_processed.json')
    bot.load_model('data/models/tfidf_model.joblib')
    return bot


@st.cache_resource
def load_w2v():
    bot = Word2VecChatbot('data/processed/dataset_processed.json')
    bot.load_model('data/models/w2v_model.joblib')
    return bot


@st.cache_resource
def load_embedding():
    bot = EmbeddingChatbot('data/processed/dataset_processed.json')
    bot.load_model('data/models/embedding_model.joblib')
    return bot


if model_sec == "TF-IDF":
    bot = load_tfidf()
    threshold = threshold  # sidebar'dan geliyor
elif model_sec == "Word2Vec":
    bot = load_w2v()
else:
    bot = load_embedding()

# ── Sohbet geçmişi ────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_question" not in st.session_state:
    st.session_state.selected_question = None

st.title("⚖️ Boşanma Hukuku Bilgi Asistanı")
st.caption("Türk Medeni Kanunu kapsamında genel hukuki bilgi sunar.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Örnek sorular ─────────────────────────────────────────────────────
ORNEK_SORULAR = [
    "Boşanma davası açmak için hangi koşullar gereklidir?",
    "Anlaşmalı boşanma ile çekişmeli boşanma arasındaki fark nedir?",
    "Boşanmada nafaka nasıl hesaplanır?",
    "Çocukların velayeti boşanmada nasıl belirlenir?",
    "Mal paylaşımı boşanmada nasıl yapılır?",
    "Zina nedeniyle boşanma davası açılabilir mi?",
    "Boşanma davası ne kadar sürer?",
]

if not st.session_state.messages:
    st.markdown("**Örnek sorular:**")
    cols = st.columns(2)
    for i, soru in enumerate(ORNEK_SORULAR):
        col = cols[i % 2]
        if col.button(soru, key=f"ornek_{i}", use_container_width=True):
            st.session_state.selected_question = soru
            st.rerun()

# ── Kullanıcı girişi ──────────────────────────────────────────────────
selected = None
if st.session_state.selected_question:
    selected = st.session_state.selected_question
    st.session_state.selected_question = None

if prompt := (selected or st.chat_input("Sorunuzu yazın...")):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Yanıt aranıyor..."):
        result = bot.respond(prompt, threshold=threshold)

    if result["skor"] == 0.0:
        cevap_metni = result["cevap"]
    else:
        cevap_metni = (
            f"{result['cevap']}\n\n"
            f"---\n"
            f"📂 **Kategori:** {result['kategori']}  \n"
            f"📖 **Kaynak:** {result['kaynak']}  \n"
            f"🎯 **Benzerlik skoru:** {result['skor']:.4f}"
        )

    with st.chat_message("assistant"):
        st.markdown(cevap_metni)

    st.session_state.messages.append(
        {"role": "assistant", "content": cevap_metni}
    )

    st.warning(
        "⚠️ Bu bilgiler genel amaçlıdır, hukuki danışmanlık yerine geçmez."
    )
