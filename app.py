import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()

# ---------------- SAYFA AYARLARI ----------------
st.set_page_config(page_title="Banka Asistanı", page_icon="🏦", layout="centered")

# ---------------- ÖZEL TASARIM (Kırmızı-Lacivert-Beyaz) ----------------

st.markdown("""
<style>
    /* Koyu lacivert arka plan */
    .stApp {
        background: linear-gradient(160deg, #0d1b2a 0%, #1b2a41 100%);
    }
    /* Çapraz filigran */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        pointer-events: none; z-index: 0;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='320' height='160'%3E%3Ctext x='0' y='80' font-family='Arial' font-size='22' font-weight='bold' fill='%23ffffff' fill-opacity='0.03' transform='rotate(-30 160 80)'%3EBanka Asistanı%3C/text%3E%3C/svg%3E");
        background-repeat: repeat;
    }
    /* Başlık bandı */
    .header-band {
        text-align: center;
        padding: 20px 0 10px 0;
        position: relative;
        z-index: 1;
    }
    .logo-row {
        display: flex; align-items: center; justify-content: center;
        gap: 14px; margin-bottom: 18px;
    }
    .logo-circle {
        background: #ffffff;
        color: #c8102e;
        width: 68px; height: 68px;
        border-radius: 50%;
        border: 3px solid #c8102e;
        display: flex; align-items: center; justify-content: center;
        font-size: 28px; font-weight: 800; letter-spacing: -1px;
        box-shadow: 0 4px 14px rgba(200,16,46,0.45);
    }
    .logo-circle span { color: #c8102e; }
    .brand-text {
        color: #e8ecf3; font-size: 22px; font-weight: 800; letter-spacing: 1px;
    }
    .main-title {
        color: #ffffff; font-size: 34px; font-weight: 800;
        letter-spacing: 2px; margin: 10px 0 6px 0;
    }
    .title-divider {
        height: 2px; width: 60%; margin: 0 auto 20px auto;
        background: linear-gradient(90deg, transparent, #c8102e, transparent);
    }
    /* Örnek soru butonları - beyaz kart */
    .stButton button {
        background-color: #ffffff;
        color: #0d1b2a;
        border: none;
        border-radius: 12px;
        padding: 16px 14px;
        font-size: 14px; font-weight: 600;
        width: 100%;
        box-shadow: 0 3px 10px rgba(0,0,0,0.3);
        transition: all 0.2s;
    }
    .stButton button:hover {
        background-color: #f0f0f0;
        transform: translateY(-2px);
        box-shadow: 0 5px 14px rgba(200,16,46,0.3);
    }
    /* Kullanıcı mesaj balonu - kırmızı */
    .user-msg {
        background-color: #c8102e;
        color: #ffffff;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0 8px auto;
        max-width: 78%; width: fit-content;
        font-size: 15px;
        box-shadow: 0 2px 8px rgba(200,16,46,0.3);
        position: relative; z-index: 1;
    }
    /* Asistan mesaj balonu - koyu */
    .bot-msg {
        background-color: #22334d;
        color: #e8ecf3;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px auto 8px 0;
        max-width: 78%; width: fit-content;
        font-size: 15px;
        border-left: 3px solid #c8102e;
        position: relative; z-index: 1;
    }
    .section-label {
        color: #c8102e; font-weight: 700; font-size: 15px;
        margin-bottom: 10px; position: relative; z-index: 1;
    }
    
</style>
""", unsafe_allow_html=True)

 

# ---------------- BAŞLIK ----------------
st.markdown("""
<div class="header-band">
    <div class="logo-row">
        <div class="logo-circle"><span>B</span>A</div>
        <div class="brand-text">DİJİTAL BANKACILIK</div>
    </div>
    <div class="main-title">BANKA ASİSTANI</div>
    <div class="title-divider"></div>
</div>
""", unsafe_allow_html=True)
# ---------------- RAG SİSTEMİ ----------------
@st.cache_resource
def load_rag_system():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    vectorstore = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.2
    )

    prompt_template = """Sen bir bankanın profesyonel müşteri hizmetleri asistanısın.

Görevin: Aşağıdaki bağlam bilgilerini kullanarak müşterinin sorusunu yanıtlamak.

Kurallar:
- Yanıtın akıcı, dilbilgisi açısından kusursuz ve anlamlı tam cümlelerden oluşmalı.
- Kibar, profesyonel ve net bir dil kullan.
- Sadece bağlamdaki bilgiyi kullan, kesinlikle bilgi uydurma.
- Eğer cevap bağlamda yoksa, şu şekilde yanıtla: "Bu konuda elimde bilgi bulunmuyor. Lütfen daha detaylı bilgi için müşteri hizmetlerimizi arayın."
- Selamlama (Merhaba vb.) kullanma; doğrudan sorunun cevabına geç.
- Yanıtı nazik ama kısa bir kapanışla bitirebilirsin, ancak her seferinde aynı kapanışı tekrarlama.

Bağlam:
{context}

Müşteri Sorusu: {question}

Yanıt:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

qa_chain = load_rag_system()

# ---------------- SOHBET GEÇMİŞİ (session state) ----------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Merhaba! 👋 Ben banka müşteri hizmetleri asistanınızım. Size nasıl yardımcı olabilirim?"}
    ]
# Sağ üstte küçük "Sohbeti Temizle" butonu
_, temizle_col = st.columns([4, 1])
with temizle_col:
    if st.button("🗑️ Temizle", type="tertiary"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Merhaba! 👋 Ben banka müşteri hizmetleri asistanınızım. Size nasıl yardımcı olabilirim?"}
        ]
        st.rerun()
# ---------------- ÖRNEK SORULAR ----------------

ornek_sorular = [
    "💳 Kartım kayboldu",
    "💰 Kredi kartı aidat ücreti nedir?",
    "📄 Hesap açma şartları nelerdir?",
    "🏠 İhtiyaç kredisi nasıl alabilirim?",
    "🔒 Şifremi unuttum",
    "🕐 Çalışma saatleri nedir?"
]
cols = st.columns(2)
secilen_soru = None
for i, soru in enumerate(ornek_sorular):
    if cols[i % 2].button(soru, key=f"ornek_{i}"):
        # Emojiyi kaldırıp sadece metni gönder
        secilen_soru = soru.split(" ", 1)[1] if " " in soru else soru

st.divider()

# ---------------- SOHBET GEÇMİŞİNİ GÖSTER ----------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">{msg["content"]}</div>', unsafe_allow_html=True)

# ---------------- KULLANICI GİRİŞİ ----------------
kullanici_girisi = st.chat_input("Sorunuzu yazın...")

# Örnek soru butonuna basıldıysa onu kullan
soru = secilen_soru if secilen_soru else kullanici_girisi

if soru:
    # Kullanıcı mesajını ekle ve göster
    st.session_state.messages.append({"role": "user", "content": soru})
    st.markdown(f'<div class="user-msg">{soru}</div>', unsafe_allow_html=True)

    # Cevap üret
    with st.spinner("Yanıt hazırlanıyor..."):
        result = qa_chain.invoke({"query": soru})
        cevap = result["result"]

    # Asistan mesajını ekle ve göster
    st.session_state.messages.append({"role": "assistant", "content": cevap})
    st.markdown(f'<div class="bot-msg">{cevap}</div>', unsafe_allow_html=True)

# ---------------- ALT BİLGİ ----------------
st.divider()
st.caption("Bu asistan yalnızca bilgi tabanındaki konularda yardımcı olabilir. "
           "Karmaşık işlemler için lütfen müşteri hizmetlerimizi arayın.")