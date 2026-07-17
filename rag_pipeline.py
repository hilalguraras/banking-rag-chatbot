import os 
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

loader = TextLoader("data/bank_faq.txt", encoding="utf-8")
documents = loader.load()

print(f"Yüklenen döküman sayısı: {len(documents)}")
print(f"Toplam karakter sayısı: {len(documents[0].page_content)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators = ["\n\n", "\n", ". "," ",""]
)

chunks = text_splitter.split_documents(documents)

print(f"\nOluşturulan parça (chunk) sayısı: {len(chunks)}")
print(f"\n--- İlk parça örneği ---")
print(chunks[0].page_content)
print(f"\n--- İkinci parça örneği ---")
print(chunks[1].page_content)

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# 3. Embedding modelini tanımla (Gemini'nin embedding modeli)
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# 4. Chunk'ları vektöre çevirip FAISS veritabanına kaydet
print("\nEmbedding oluşturuluyor ve FAISS veritabanı kuruluyor...")
vectorstore = FAISS.from_documents(chunks, embeddings)

# 5. Veritabanını diske kaydet (her seferinde yeniden oluşturmamak için)
vectorstore.save_local("faiss_index")
print("FAISS veritabanı başarıyla oluşturuldu ve 'faiss_index' klasörüne kaydedildi!")

# 6. Hızlı bir test - benzerlik araması yapalım
print("\n--- Test: 'kredi kartı aidatı ne kadar?' sorusuna en yakın parçalar ---")
test_query = "kredi kartı aidatı ne kadar?"
similar_docs = vectorstore.similarity_search(test_query, k=2)

for i, doc in enumerate(similar_docs, 1):
    print(f"\n[{i}. en yakın parça]")
    print(doc.page_content[:200])


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# 7. LLM'i tanımlama 
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3 # daha tutarlı, uydurmasız cevaplar
)

# 8. Özel bir prompt şablonu - chatbot'un davranışını belirler 
prompt_template = """Sen bir bankanın müşteri hizmetleri asistanısın. 
Aşağıdaki bağlam bilgilerini kullanarak müşterinin sorusunu Türkçe, kibar ve net bir şekilde yanıtla.
Eğer cevap bağlamda yoksa, "Bu konuda elimde bilgi bulunmuyor, lütfen müşteri hizmetlerimizi arayın." de.
Bağlamda olmayan bilgi uydurma.

Bağlam:
{context}

Soru: {question}

Cevap:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# 9. RAG zincirini oluştur (retrieval + generation)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),  # en yakın 2 parçayı getir
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

# 10. Test edelim
print("\n" + "="*50)
print("RAG CHATBOT TEST")
print("="*50)

test_questions = [
    "Kredi kartı aidatı ne kadar?",
    "Hesap açmak için ne gerekli?",
    "Kartım çalındı, ne yapmalıyım?",
    "Bitcoin alabilir miyim?"  # bilgi tabanında olmayan bir soru - "bilmiyorum" demeli
]

for q in test_questions:
    print(f"\n SORU: {q}")
    result = qa_chain.invoke({"query": q})
    print(f" CEVAP: {result['result']}")