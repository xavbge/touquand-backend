from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import replicate
import os
import io
import json
import pycountry
import re
import traceback
import base64
from dotenv import load_dotenv
import requests
from urllib.parse import quote
from typing import Optional, Dict, Any

load_dotenv()

app = FastAPI(
    title="Touquand - GPT-4o Vision API",
    description="API d'extraction d'informations d'affiches via GPT-4o Vision",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration du modèle
GPT4O_MODEL = "openai/gpt-4o"
MAX_RETRIES = 2
RETRY_DELAY = 3
SERP_API_KEY = os.getenv("SERP_API_KEY")


def image_to_base64(image_bytes: bytes) -> str:
    """Convertit des bytes d'image en base64 data URL."""
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_image}"

def search_web(query: str) -> Optional[str]:
    """Recherche rapide sur le web pour compléter les infos manquantes."""
    if not SERP_API_KEY:
        print("⚠️ Pas de clé SERP_API_KEY, recherche web désactivée.")
        return None
    
    url = f"https://serpapi.com/search.json?q={quote(query)}&hl=fr&gl=fr&api_key={SERP_API_KEY}"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            # Recherche d’un extrait avec prix ou intervalle
            snippets = " ".join(
                [r.get("snippet", "") for r in data.get("organic_results", [])]
            )
            price_match = re.search(r"(\d{1,3}(?:[.,]\d{2})?)\s?€", snippets)
            if price_match:
                prices = re.findall(r"(\d{1,3}(?:[.,]\d{2})?)\s?€", snippets)
                if prices:
                    prices = sorted([float(p.replace(",", ".")) for p in prices])
                    if len(prices) > 1:
                        return f"Estimation comprise entre {int(prices[0])}€ et {int(prices[-1])}€ (EUR - France)"
                    return f"Estimation d’environ {int(prices[0])}€ (EUR - France)"
        return None
    except Exception as e:
        print(f"❌ Erreur recherche web : {e}")
        return None



def clean_json_string(text: str) -> str:
    """Nettoie une chaîne pour faciliter le parsing JSON."""
    # Supprime les blocs markdown
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    return text.strip()


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extrait un objet JSON d'un texte brut avec plusieurs stratégies.
    """
    original_text = text
    text = clean_json_string(text)
    
    print(f"🔍 Début du texte à parser: {text[:150]}...")
    
    # Stratégie 1: Parser direct
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Stratégie 2: Extraire le premier objet JSON valide
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Stratégie 3: Recherche permissive + réparation guillemets
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Réparer les guillemets simples
            try:
                repaired = re.sub(r"'(\w+)':", r'"\1":', json_str)
                repaired = re.sub(r":\s*'([^']*)'", r': "\1"', repaired)
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
    
    # Échec total
    print("❌ Parsing JSON échoué")
    return {
        "parsing_failed": True,
        "raw_response": original_text[:500],
        "error": "Impossible de parser le JSON"
    }


def validate_extracted_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Valide et normalise les données extraites."""
    required_fields = ["titre", "date", "lieu", "prix", "categorie", "lien_billetterie", "description"]
    
    # Si le parsing a échoué
    if data.get("parsing_failed"):
        return {
            "titre": "Erreur d'extraction",
            "date": "Non détecté",
            "lieu": "Non détecté",
            "prix": "Non détecté",
            "categorie": "Non détecté",
            "lien_billetterie": "Non détecté",
            "description": data.get("raw_response", "Erreur lors de l'extraction"),
            "_parsing_error": True
        }
    
    # Normalisation des champs
    for field in required_fields:
        if field not in data or data[field] is None or str(data[field]).strip() == "":
            data[field] = "Non détecté"
    
    return data

def detect_currency_from_location(location: str) -> Dict[str, str]:
    """
    Déduit la devise et le pays à partir du nom du lieu.
    Utilise des correspondances simples + fallback via pycountry.
    """
    location_lower = location.lower()
    mapping = {
        "france": ("EUR", "France"),
        "paris": ("EUR", "France"),
        "lyon": ("EUR", "France"),
        "marseille": ("EUR", "France"),
        "bruxelles": ("EUR", "Belgique"),
        "belgique": ("EUR", "Belgique"),
        "montréal": ("CAD", "Canada"),
        "québec": ("CAD", "Canada"),
        "canada": ("CAD", "Canada"),
        "londres": ("GBP", "Royaume-Uni"),
        "uk": ("GBP", "Royaume-Uni"),
        "angleterre": ("GBP", "Royaume-Uni"),
        "new york": ("USD", "États-Unis"),
        "los angeles": ("USD", "États-Unis"),
        "usa": ("USD", "États-Unis"),
        "états-unis": ("USD", "États-Unis"),
        "berlin": ("EUR", "Allemagne"),
        "espagne": ("EUR", "Espagne"),
        "italie": ("EUR", "Italie"),
        "rome": ("EUR", "Italie"),
    }

    # Vérifie les correspondances connues
    for key, (currency, country) in mapping.items():
        if key in location_lower:
            return {"currency": currency, "country": country}

    # Fallback si non trouvé : détecte le pays dans la chaîne (pycountry)
    for country in pycountry.countries:
        if country.name.lower() in location_lower:
            currency = "EUR"
            if hasattr(country, "alpha_2"):
                if country.alpha_2 in ["US"]:
                    currency = "USD"
                elif country.alpha_2 in ["GB"]:
                    currency = "GBP"
                elif country.alpha_2 in ["CA"]:
                    currency = "CAD"
                elif country.alpha_2 in ["CH"]:
                    currency = "CHF"
            return {"currency": currency, "country": country.name}

    # Par défaut : France / EUR
    return {"currency": "EUR", "country": "France"}


def search_web_for_price(event_name: str, location: str = "", category: str = "concert") -> Optional[str]:
    """Recherche une estimation de prix sur le web via SerpAPI."""
    serp_api_key = os.getenv("SERP_API_KEY")
    if not serp_api_key:
        print("⚠️ Pas de SERP_API_KEY dans .env — recherche web désactivée.")
        return None

    query = f"{event_name} {location} {category} prix billets"
    url = f"https://serpapi.com/search.json?q={quote(query)}&hl=fr&gl=fr&api_key={serp_api_key}"

    try:
        res = requests.get(url, timeout=10)
        if res.status_code != 200:
            return None

        data = res.json()
        snippets = " ".join(
            r.get("snippet", "") for r in data.get("organic_results", [])
        )

        # Extraire tous les montants en euros
        prices = re.findall(r"(\d{1,3}(?:[.,]\d{2})?)\s?€", snippets)
        if prices:
            prices = sorted([float(p.replace(",", ".")) for p in prices])
            if len(prices) >= 2:
                return f"Estimation comprise entre {int(prices[0])}€ et {int(prices[-1])}€ (EUR - France)"
            return f"Estimation d’environ {int(prices[0])}€ (EUR - France)"
        return None
    except Exception as e:
        print(f"❌ Erreur recherche web : {e}")
        return None



def generate_system_prompt() -> str:
    """Génère le prompt système pour GPT-4o."""
    return """Tu es un expert en analyse visuelle d'affiches d'événements culturels.

MISSION:
Extraire TOUTES les informations visibles de l'affiche et les retourner dans un format JSON structuré.

FORMAT DE RÉPONSE OBLIGATOIRE:
Tu dois répondre avec un objet JSON valide et UNIQUEMENT du JSON. Pas de texte avant, pas de texte après, pas de markdown.

Structure JSON attendue:
{
  "titre": "Titre exact de l'événement (artistes, nom du spectacle)",
  "date": "Date complète avec jour, mois, année et horaire si visible",
  "lieu": "Nom de la salle/lieu ET ville si visible",
  "prix": "Prix ou fourchette de prix si indiqué",
  "categorie": "Type d'événement: concert, théâtre, festival, exposition, conférence, sport, etc.",
  "lien_billetterie": "URL ou site web de billetterie si visible",
  "description": "Résumé en 2-3 phrases avec informations importantes: têtes d'affiche, particularités, invités spéciaux, etc."
}

RÈGLES D'EXTRACTION:
1. Titre: Extrais le nom principal de l'événement en MAJUSCULES si c'est écrit ainsi
2. Date: Cherche partout (en haut, en bas, en petit) - format: "Jeudi 2 juillet 2026"
3. Lieu: Nom du lieu + ville (ex: "Stade de France - Paris")
4. Prix: Si plusieurs tarifs, indique la fourchette (ex: "45€ - 85€")
5. Catégorie: Déduis du contenu visuel si non explicite
6. Lien: Cherche les URLs, sites web, QR codes avec texte associé
7. Description: Mentionne les artistes principaux, supports, particularités

RÈGLES TECHNIQUES:
- Si une info est absente: écris "Non détecté"
- Utilise UNIQUEMENT des guillemets doubles "
- Ne mets AUCUN texte explicatif
- Pas de ```json ni de code block
- Commence directement par { et termine par }

IMPORTANT: Lis TOUTE l'affiche, y compris les petits textes en bas et sur les côtés."""


def generate_user_prompt() -> str:
    """Génère le prompt utilisateur."""
    return """Analyse cette affiche d'événement.

Identifie:
- Le nom de l'événement/des artistes (en haut ou au centre)
- La date et l'heure (cherche partout, même en petit)
- Le lieu exact (nom de la salle + ville)
- Le prix si indiqué
- Le type d'événement
- Les sites web ou liens de billetterie
- Les détails importants (invités, particularités)

Retourne le JSON structuré (commence par {, termine par }):"""


async def analyze_with_gpt4o(image_input: Any, image_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyse l'image avec GPT-4o Vision en un seul appel.
    """
    system_prompt = generate_system_prompt()
    user_prompt = generate_user_prompt()
    
    # Préparer le contenu du message
    content = []
    
    # Ajouter l'image
    if image_url:
        # Si c'est une URL externe
        content.append({
            "type": "image_url",
            "image_url": {"url": image_url}
        })
    elif isinstance(image_input, io.BytesIO):
        # Si c'est un fichier uploadé, le convertir en base64
        image_input.seek(0)
        image_bytes = image_input.read()
        base64_image = image_to_base64(image_bytes)
        content.append({
            "type": "image_url",
            "image_url": {"url": base64_image}
        })
    
    # Ajouter le texte
    content.append({
        "type": "text",
        "text": user_prompt
    })
    
    # Construire l'input pour Replicate
    gpt_input = {
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": content
            }
        ],
        "max_tokens": 1000,
        "temperature": 0.1,  # Très basse pour cohérence
        "top_p": 0.9
    }
    
    print("🧠 Analyse en cours via GPT-4o Vision...")
    
    try:
        # Appel à GPT-4o - récupère le stream complet
        output = replicate.run(GPT4O_MODEL, input=gpt_input)
        
        # GPT-4o retourne un itérateur de tokens - on doit les joindre
        if hasattr(output, '__iter__') and not isinstance(output, str):
            response_text = ''.join(str(token) for token in output)
        else:
            response_text = str(output)
        
        response_text = response_text.strip()
        print(f"📝 Réponse GPT-4o (200 premiers chars): {response_text[:200]}")
        
        # Extraction du JSON
        extracted_data = extract_json_from_text(response_text)
        validated_data = validate_extracted_data(extracted_data)
        
        return validated_data
    
    except Exception as e:
        print(f"❌ Erreur GPT-4o: {e}")
        raise


@app.post("/analyze-image")
async def analyze_image(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None)
):
    """
    Analyse une image d'affiche et extrait les informations structurées en un seul appel.
    
    **Architecture simplifiée:**
    - Un seul modèle: GPT-4o Vision
    - Vision + LLM en une seule étape
    - Extraction directe du JSON structuré
    
    **Paramètres:**
    - file: Fichier image uploadé
    - image_url: URL d'une image en ligne
    
    **Retour:**
    - success: Statut de l'opération
    - structured_data: Données structurées en JSON
    - raw_response: Réponse brute (optionnel, pour debug)
    """
    try:
        # Validation de l'entrée
        image_input = None
        url_for_analysis = None
        
        if file:
            print(f"⬆️ Fichier uploadé : {file.filename}")
            image_bytes = await file.read()
            image_input = io.BytesIO(image_bytes)
            image_input.name = file.filename
        elif image_url:
            print(f"⬆️ URL fournie : {image_url}")
            url_for_analysis = image_url
        else:
            raise HTTPException(
                status_code=400,
                detail="Aucune image fournie. Veuillez fournir un fichier ou une URL."
            )

        # Analyse avec GPT-4o Vision (un seul appel)
        structured_data = await analyze_with_gpt4o(image_input, url_for_analysis)
        
        # 🌍 Complément automatique si certains champs sont "Non détecté"
        if structured_data.get("prix") == "Non détecté":
            titre = structured_data.get("titre", "")
            lieu = structured_data.get("lieu", "")
            categorie = structured_data.get("categorie", "concert")
            estimation = search_web_for_price(titre, lieu, categorie)
            if estimation:
                structured_data["prix"] = estimation

        
        parsing_error = structured_data.pop("_parsing_error", False)
        
        print("✅ Analyse terminée avec succès")

        # Réponse finale
        return JSONResponse({
            "success": True,
            "structured_data": structured_data,
            "warnings": {
                "parsing_error": parsing_error
            },
            "metadata": {
                "model": "GPT-4o Vision",
                "architecture": "Single-step vision + LLM"
            }
        })

    except HTTPException as http_exc:
        raise http_exc
    
    except Exception as e:
        print(f"❌ Erreur critique : {e}")
        traceback.print_exc()
        return JSONResponse(
            {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            },
            status_code=500
        )


@app.get("/")
async def root():
    """Point d'entrée de l'API."""
    return {
        "service": "Touquand - GPT-4o Vision API",
        "status": "operational",
        "version": "3.0.0",
        "features": [
            "Single-step vision analysis",
            "GPT-4o multimodal model",
            "Direct JSON extraction",
            "Supports URL and file upload"
        ],
        "endpoints": {
            "analyze": "/analyze-image",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Vérification de santé de l'API."""
    replicate_token = os.getenv("REPLICATE_API_TOKEN")
    return {
        "status": "healthy",
        "replicate_configured": bool(replicate_token),
        "model": GPT4O_MODEL
    }


if __name__ == "__main__":
    import uvicorn
    
    if not os.getenv("REPLICATE_API_TOKEN"):
        print("❌ ERREUR : REPLICATE_API_TOKEN non défini dans .env")
        print("💡 Créez un fichier .env avec : REPLICATE_API_TOKEN=votre_token")
        exit(1)
    
    print("=" * 60)
    print("🚀 Serveur Touquand-GPT-4o Vision API v3.0")
    print("=" * 60)
    print("📍 URL : http://127.0.0.1:8000")
    print("📚 Documentation : http://127.0.0.1:8000/docs")
    print("🔑 Token Replicate : Chargé ✓")
    print("🤖 Modèle : GPT-4o (Vision + LLM en un appel)")
    print("=" * 60)
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)