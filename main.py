from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import io
import json
import pycountry
import re
import traceback
import requests
from urllib.parse import quote
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Imports spécifiques pour Gemini et le traitement d'image
import google.generativeai as genai
from PIL import Image

load_dotenv()

app = FastAPI(
    title="Touquand - Gemini Flash API",
    description="API d'extraction d'informations d'affiches via Google Gemini 1.5 Flash (Gratuit)",
    version="3.1.2"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === CONFIGURATION DES CLÉS API ===
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")

if not GOOGLE_API_KEY:
    print("⚠️  ATTENTION : GOOGLE_API_KEY manquante ! L'analyse échouera.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # --- DIAGNOSTIC DÉMARRAGE : LISTER LES MODÈLES ---
    print("🔎 VÉRIFICATION DES MODÈLES DISPONIBLES...")
    try:
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
                print(f"   ✅ Trouvé : {m.name}")
        
        if not available_models:
            print("   ⚠️ Aucun modèle 'generateContent' trouvé. Vérifiez votre clé API.")
    except Exception as e:
        print(f"   ❌ Erreur lors du listing des modèles : {e}")
    print("------------------------------------------------")

# On remet le nom standard. Si ça échoue, regardez les logs "VÉRIFICATION" ci-dessus.
GEMINI_MODEL_NAME = 'gemini-1.5-flash'


# === FONCTIONS UTILITAIRES ===

def clean_json_string(text: str) -> str:
    """Nettoie une chaîne pour faciliter le parsing JSON."""
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    return text.strip()

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extrait un objet JSON d'un texte brut."""
    original_text = text
    text = clean_json_string(text)
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Regex de secours
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
            
    # Tentative de réparation des guillemets
    try:
        repaired = re.sub(r"'(\w+)':", r'"\1":', text)
        repaired = re.sub(r":\s*'([^']*)'", r': "\1"', repaired)
        return json.loads(repaired)
    except Exception:
        pass

    print(f"❌ Échec parsing JSON. Texte reçu : {original_text[:200]}...")
    return {
        "parsing_failed": True,
        "raw_response": original_text[:500],
        "error": "Impossible de parser le JSON"
    }

def validate_extracted_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Valide et normalise les données."""
    required_fields = ["titre", "date", "lieu", "prix", "categorie", "lien_billetterie", "description"]
    
    if data.get("parsing_failed"):
        return {k: "Non détecté" for k in required_fields}
    
    for field in required_fields:
        if field not in data or not data[field] or str(data[field]).strip() == "":
            data[field] = "Non détecté"
    return data

def detect_currency_from_location(location: str) -> Dict[str, str]:
    """Déduit la devise via Pycountry."""
    if not location or location == "Non détecté":
        return {"currency": "EUR", "country": "France"}
        
    location_lower = location.lower()
    mapping = {
        "paris": ("EUR", "France"), "lyon": ("EUR", "France"), "france": ("EUR", "France"),
        "bruxelles": ("EUR", "Belgique"), "belgique": ("EUR", "Belgique"),
        "montréal": ("CAD", "Canada"), "canada": ("CAD", "Canada"),
        "londres": ("GBP", "Royaume-Uni"), "uk": ("GBP", "Royaume-Uni"),
        "usa": ("USD", "États-Unis"), "new york": ("USD", "États-Unis")
    }
    
    for key, (currency, country) in mapping.items():
        if key in location_lower:
            return {"currency": currency, "country": country}

    for country in pycountry.countries:
        if country.name.lower() in location_lower:
            currency = "EUR"
            if hasattr(country, "alpha_2"):
                if country.alpha_2 == "US": currency = "USD"
                elif country.alpha_2 == "GB": currency = "GBP"
                elif country.alpha_2 == "CA": currency = "CAD"
                elif country.alpha_2 == "CH": currency = "CHF"
            return {"currency": currency, "country": country.name}

    return {"currency": "EUR", "country": "France"}

def search_web_for_price(event_name: str, location: str = "", category: str = "concert") -> Optional[str]:
    """Recherche une estimation de prix sur le web via SerpAPI."""
    if not SERP_API_KEY:
        return None

    print(f"🔍 Recherche Web pour : {event_name} à {location}")
    query = f"{event_name} {location} {category} prix billets"
    url = f"https://serpapi.com/search.json?q={quote(query)}&hl=fr&gl=fr&api_key={SERP_API_KEY}"

    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            snippets = " ".join(r.get("snippet", "") for r in data.get("organic_results", []))
            prices = re.findall(r"(\d{1,3}(?:[.,]\d{2})?)\s?€", snippets)
            if prices:
                prices = sorted([float(p.replace(",", ".")) for p in prices])
                if len(prices) >= 2:
                    return f"Estimation : {int(prices[0])}€ - {int(prices[-1])}€ (Web)"
                return f"Estimation : ~{int(prices[0])}€ (Web)"
    except Exception as e:
        print(f"❌ Erreur recherche web : {e}")
    return None

async def download_image(url: str) -> bytes:
    """Télécharge une image depuis une URL."""
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.content


# === LOGIQUE GEMINI ===

def generate_gemini_prompt() -> str:
    return """
    Tu es un expert en analyse visuelle d'affiches d'événements.
    Analyse cette image et extrais les informations suivantes au format JSON strict.

    Champs requis :
    - "titre": Titre de l'événement.
    - "date": Date complète (Jour, Mois, Année, Heure).
    - "lieu": Lieu exact et ville.
    - "prix": Le prix ou "Gratuit" ou "Non détecté".
    - "categorie": Type d'événement.
    - "lien_billetterie": URL si visible.
    - "description": Résumé court.

    RÉPONSE : UNIQUEMENT LE JSON.
    """

async def analyze_with_gemini(image_bytes: bytes) -> Dict[str, Any]:
    """Envoie l'image à Google Gemini Flash."""
    try:
        # Instanciation du modèle
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        
        image = Image.open(io.BytesIO(image_bytes))
        
        print(f"🧠 Envoi à {GEMINI_MODEL_NAME}...")
        response = model.generate_content([generate_gemini_prompt(), image])
        
        print(f"📝 Réponse reçue (début): {response.text[:100]}")
        
        raw_data = extract_json_from_text(response.text)
        return validate_extracted_data(raw_data)

    except Exception as e:
        print(f"❌ Erreur Gemini : {e}")
        raise e


# === ROUTES API ===

@app.post("/analyze-image")
async def analyze_image(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None)
):
    try:
        image_bytes = None
        if file:
            print(f"⬆️ Fichier : {file.filename}")
            image_bytes = await file.read()
        elif image_url:
            print(f"⬆️ URL : {image_url}")
            image_bytes = await download_image(image_url)
        else:
            raise HTTPException(status_code=400, detail="Aucune image fournie.")

        # Analyse
        structured_data = await analyze_with_gemini(image_bytes)

        # Fallback Prix
        if structured_data.get("prix") in ["Non détecté", None, ""]:
            print("🔍 Prix manquant, tentative Web...")
            est = search_web_for_price(
                structured_data.get("titre", ""), 
                structured_data.get("lieu", ""), 
                structured_data.get("categorie", "")
            )
            if est: structured_data["prix"] = est

        # Meta devise
        structured_data["_meta"] = detect_currency_from_location(structured_data.get("lieu", ""))

        return JSONResponse({
            "success": True,
            "structured_data": structured_data,
            "metadata": {"model": GEMINI_MODEL_NAME}
        })

    except Exception as e:
        print(f"❌ Erreur critique : {e}")
        traceback.print_exc()
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/")
def root():
    return {"status": "online", "model": GEMINI_MODEL_NAME}

if __name__ == "__main__":
    import uvicorn
    if not os.getenv("GOOGLE_API_KEY"): print("⚠️ GOOGLE_API_KEY manquante")
    uvicorn.run(app, host="0.0.0.0", port=8000)