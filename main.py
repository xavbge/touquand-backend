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
    version="3.1.0"
)

# Configuration CORS (Indispensable pour que le mobile puisse parler au backend)
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
    # Configuration de Gemini
    genai.configure(api_key=GOOGLE_API_KEY)

# Modèle utilisé : Gemini 1.5 Flash (Rapide et Gratuit)
GEMINI_MODEL_NAME = 'gemini-1.5-flash'


# === FONCTIONS UTILITAIRES (Conservées de ton ancien code) ===

def clean_json_string(text: str) -> str:
    """Nettoie une chaîne pour faciliter le parsing JSON (enlève le markdown)."""
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    return text.strip()

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extrait un objet JSON d'un texte brut avec plusieurs stratégies de secours."""
    original_text = text
    text = clean_json_string(text)
    
    # Stratégie 1 : Parsing direct
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Stratégie 2 : Regex pour trouver le bloc JSON
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Stratégie 3 : Tentative de réparation des guillemets simples (de ton code original)
    try:
        repaired = re.sub(r"'(\w+)':", r'"\1":', text)
        repaired = re.sub(r":\s*'([^']*)'", r': "\1"', repaired)
        return json.loads(repaired)
    except Exception:
        pass

    # Échec
    print(f"❌ Échec parsing JSON. Texte reçu : {original_text[:200]}...")
    return {
        "parsing_failed": True,
        "raw_response": original_text[:500],
        "error": "Impossible de parser le JSON"
    }

def validate_extracted_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Valide et normalise les données extraites (remplit les vides)."""
    required_fields = ["titre", "date", "lieu", "prix", "categorie", "lien_billetterie", "description"]
    
    if data.get("parsing_failed"):
        return {k: "Non détecté" for k in required_fields}
    
    for field in required_fields:
        if field not in data or not data[field] or str(data[field]).strip() == "":
            data[field] = "Non détecté"
    return data

def detect_currency_from_location(location: str) -> Dict[str, str]:
    """(Ton code original) Déduit la devise via Pycountry."""
    if not location or location == "Non détecté":
        return {"currency": "EUR", "country": "France"}
        
    location_lower = location.lower()
    # Mapping manuel rapide
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

    # Fallback Pycountry
    for country in pycountry.countries:
        if country.name.lower() in location_lower:
            currency = "EUR" # Défaut Europe
            if hasattr(country, "alpha_2"):
                if country.alpha_2 == "US": currency = "USD"
                elif country.alpha_2 == "GB": currency = "GBP"
                elif country.alpha_2 == "CA": currency = "CAD"
                elif country.alpha_2 == "CH": currency = "CHF"
            return {"currency": currency, "country": country.name}

    return {"currency": "EUR", "country": "France"}

def search_web_for_price(event_name: str, location: str = "", category: str = "concert") -> Optional[str]:
    """(Ton code original) Recherche une estimation de prix sur le web via SerpAPI."""
    if not SERP_API_KEY:
        print("⚠️ Pas de SERP_API_KEY, recherche web désactivée.")
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
    """Télécharge une image depuis une URL (si l'app envoie une URL)."""
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.content


# === LOGIQUE GEMINI ===

def generate_gemini_prompt() -> str:
    return """
    Tu es un expert en analyse visuelle d'affiches d'événements culturels.
    Ta mission est d'extraire les informations visibles de l'affiche et les retourner dans un format JSON strict.

    Champs requis dans le JSON :
    - "titre": Titre exact de l'événement.
    - "date": Date complète avec jour, mois, année et heure (ex: "Samedi 12 Juillet 2025 à 20h"). Cherche partout.
    - "lieu": Nom de la salle et ville.
    - "prix": Le prix ou "Gratuit". Si tu vois plusieurs prix, mets la fourchette. Si rien n'est indiqué, mets "Non détecté".
    - "categorie": Type d'événement (Concert, Théâtre, Sport, Brocante, Conférence...).
    - "lien_billetterie": Site web ou URL visible.
    - "description": Résumé court en 2 phrases (artistes, contexte).

    RÈGLES IMPORTANTES :
    1. Réponds UNIQUEMENT avec le JSON valide.
    2. Ne mets pas de balises markdown (```json).
    3. Si une info est introuvable, écris "Non détecté".
    """

async def analyze_with_gemini(image_bytes: bytes) -> Dict[str, Any]:
    """Envoie l'image à Google Gemini Flash et retourne les données structurées."""
    try:
        # 1. Préparer le modèle
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        
        # 2. Charger l'image avec PIL (Gemini demande un objet PIL pour la vision)
        image = Image.open(io.BytesIO(image_bytes))
        
        # 3. Le prompt
        prompt = generate_gemini_prompt()
        
        print("🧠 Envoi à Gemini 1.5 Flash...")
        # Appel à l'API (multimodal : texte + image)
        response = model.generate_content([prompt, image])
        
        # 4. Récupérer le texte de réponse
        response_text = response.text
        print(f"📝 Réponse brute Gemini : {response_text[:100]}...")
        
        # 5. Parser et valider le JSON
        raw_data = extract_json_from_text(response_text)
        validated_data = validate_extracted_data(raw_data)
        
        return validated_data

    except Exception as e:
        print(f"❌ Erreur lors de l'appel Gemini : {e}")
        raise e


# === ROUTES API ===

@app.post("/analyze-image")
async def analyze_image(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None)
):
    """
    Route unique qui accepte soit un fichier uploadé, soit une URL d'image.
    Utilise Gemini 1.5 Flash (Gratuit).
    """
    try:
        image_bytes = None
        
        # 1. Récupération de l'image
        if file:
            print(f"⬆️ Fichier uploadé : {file.filename}")
            image_bytes = await file.read()
        elif image_url:
            print(f"⬆️ URL fournie : {image_url}")
            image_bytes = await download_image(image_url)
        else:
            raise HTTPException(status_code=400, detail="Aucune image fournie. Envoyez un fichier ou une image_url.")

        # 2. Analyse Gemini
        structured_data = await analyze_with_gemini(image_bytes)

        # 3. Complément automatique (Prix via Web Search) si manquant
        if structured_data.get("prix") in ["Non détecté", None, ""]:
            print("🔍 Prix non trouvé par l'IA, tentative de recherche Web...")
            titre = structured_data.get("titre", "")
            lieu = structured_data.get("lieu", "")
            categorie = structured_data.get("categorie", "événement")
            
            estimation = search_web_for_price(titre, lieu, categorie)
            if estimation:
                structured_data["prix"] = estimation
                print(f"💰 Prix trouvé sur le web : {estimation}")
        
        # 4. Ajout info devise (pour ton front-end si besoin)
        currency_info = detect_currency_from_location(structured_data.get("lieu", ""))
        structured_data["_meta"] = currency_info

        # 5. Réponse finale
        return JSONResponse({
            "success": True,
            "structured_data": structured_data,
            "metadata": {"model": GEMINI_MODEL_NAME}
        })

    except Exception as e:
        print(f"❌ Erreur critique serveur : {e}")
        traceback.print_exc()
        return JSONResponse(
            {"success": False, "error": str(e)},
            status_code=500
        )

@app.get("/")
def root():
    return {
        "service": "Touquand - Gemini API",
        "status": "operational",
        "model": GEMINI_MODEL_NAME
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "google_key_present": bool(GOOGLE_API_KEY)}

if __name__ == "__main__":
    import uvicorn
    # Vérification simple au démarrage
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️  ATTENTION : Variable GOOGLE_API_KEY manquante !")
    
    print(f"🚀 Serveur lancé sur [http://0.0.0.0:8000](http://0.0.0.0:8000) (Mode: {GEMINI_MODEL_NAME})")
    uvicorn.run(app, host="0.0.0.0", port=8000)