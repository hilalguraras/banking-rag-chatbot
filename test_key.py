import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if api_key:
    print("API anahtarı başarıyla yüklendi! İlk 10 karakter:", api_key[:10] + "...")
else:
    print("HATA: API anahtarı bulunamadı. .env dosyasını kontrol edin.")