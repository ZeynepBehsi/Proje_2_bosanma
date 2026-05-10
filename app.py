import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from src.tfidf_model import TFIDFChatbot
from src.w2v_model import Word2VecChatbot
from src.embedding_model import EmbeddingChatbot
from src.rag_model import RAGChatbot

st.set_page_config(
    page_title="Boşanma Hukuku Asistanı",
    page_icon="⚖️",
    layout="centered"
)

# ── Model yükleme (cache) ─────────────────────────────────
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

@st.cache_resource
def load_rag():
    bot = RAGChatbot(
        'data/processed/dataset_processed.json',
        'data/models/embedding_model.joblib'
    )
    return bot

# ── Session state başlat ──────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_model" not in st.session_state:
    st.session_state.current_model = None

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Ayarlar")
    model_sec = st.radio(
        "Model",
        ["TF-IDF", "Word2Vec", "Embedding", "RAG + Groq (en akıllı)"]
    )
    threshold = st.slider("Eşik skoru", 0.05, 0.50, 0.15, 0.05)

    if model_sec == "TF-IDF":
        st.info("⚡ Hızlı — Anahtar kelime eşleşmesi\n\nKategori doğruluğu: %60")
    elif model_sec == "Word2Vec":
        st.info("🧠 Orta — Anlam tabanlı eşleşme\n\nKategori doğruluğu: %58")
    elif model_sec == "Embedding":
        st.success("🎯 İyi — Çok dilli embedding\n\nKategori doğruluğu: %76")
    else:
        st.success("🤖 En akıllı — Embedding + Groq LLM\n\nDoğal Türkçe cevap üretir")

    if st.button("🗑️ Sohbeti temizle"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Bu uygulama yalnızca genel bilgi amaçlıdır. "
               "Hukuki danışmanlık yerine geçmez.")

# ── Model değişince geçmişi sıfırla ──────────────────────
if st.session_state.current_model != model_sec:
    st.session_state.messages = []
    st.session_state.current_model = model_sec

# ── Model yükle ───────────────────────────────────────────
is_rag = False
if model_sec == "TF-IDF":
    bot = load_tfidf()
elif model_sec == "Word2Vec":
    bot = load_w2v()
elif model_sec == "Embedding":
    bot = load_embedding()
else:
    bot = load_rag()
    is_rag = True

# ── Sayfa başlığı ─────────────────────────────────────────
st.title("⚖️ Boşanma Hukuku Bilgi Asistanı")
st.caption("Türk Medeni Kanunu kapsamında genel hukuki bilgi sunar.")

# ── Örnek sorular ─────────────────────────────────────────
if len(st.session_state.messages) == 0:
    st.markdown("**💡 Örnek sorular:**")
    ornek_sorular = [
        "Boşanma davası açmak için hangi koşullar gereklidir?",
        "Anlaşmalı boşanma ile çekişmeli boşanma arasındaki fark nedir?",
        "Boşanmada nafaka nasıl hesaplanır?",
        "Çocukların velayeti boşanmada nasıl belirlenir?",
        "Mal paylaşımı boşanmada nasıl yapılır?",
        "Zina nedeniyle boşanma davası açılabilir mi?",
        "Boşanma davası ne kadar sürer?",
    ]
    cols = st.columns(2)
    for i, soru in enumerate(ornek_sorular):
        col = cols[i % 2]
        if col.button(soru, key=f"ornek_{i}", use_container_width=True):
            st.session_state.messages.append(
                {"role": "user", "content": soru}
            )
            st.rerun()

# ── Geçmiş mesajları göster ───────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Kullanıcı girişi (SAYFA ALTINDA SABİT) ───────────────
prompt = st.chat_input("Sorunuzu yazın...")

if prompt:
    # Kullanıcı mesajını ekle ve göster
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Cevap üret
    with st.chat_message("assistant"):
        with st.spinner("Yanıt aranıyor..."):
            if is_rag:
                result = bot.respond(prompt)
            else:
                result = bot.respond(prompt, threshold=threshold)

        # Cevabı formatla
        if is_rag:
            if result["rag"]:
                cevap_metni = result["cevap"]
                if result["kaynaklar"]:
                    cevap_metni += "\n\n---\n"
                    for k in result["kaynaklar"]:
                        cevap_metni += (
                            f"📖 **{k['kaynak']}** ({k['kategori']}) "
                            f"— skor: {k['skor']:.4f}\n"
                        )
            else:
                cevap_metni = result["cevap"]
        else:
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

        st.markdown(cevap_metni)
        st.warning("⚠️ Bu bilgiler genel amaçlıdır, "
                   "hukuki danışmanlık yerine geçmez.")

    # Cevabı geçmişe ekle
    st.session_state.messages.append(
        {"role": "assistant", "content": cevap_metni}
    )
