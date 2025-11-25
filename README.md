# 🎨 Touquand - GPT-4o Vision API

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-009688)
![GPT-4o](https://img.shields.io/badge/Model-GPT--4o-green)

**Touquand** est une API REST performante conçue pour extraire automatiquement des informations structurées à partir d'affiches d'événements (concerts, théâtres, festivals).

Elle combine la puissance de **GPT-4o Vision** (via Replicate) pour l'analyse visuelle et utilise **SerpAPI** comme solution de secours pour estimer les prix via le web lorsque ceux-ci ne sont pas indiqués sur l'affiche.

## 🚀 Fonctionnalités Clés

- **📸 Analyse Visuelle IA** : Utilise `openai/gpt-4o` pour lire le texte, comprendre le contexte et extraire les détails d'une image.
- **📄 Sortie JSON Structurée** : Extrait automatiquement : Titre, Date, Lieu, Prix, Catégorie, Lien billetterie et Description.
- **🌍 Enrichissement Web** : Si le prix est manquant, l'API effectue une recherche intelligente sur le web pour fournir une estimation.
- **flexible** : Accepte l'upload de fichiers (`file`) ou l'envoi d'URLs (`image_url`).
- **🛡️ Parsing Robuste** : Nettoyage automatique des réponses JSON de l'IA pour garantir la stabilité de l'API.

## 🛠️ Prérequis

- Python 3.8 ou supérieur
- Un compte [Replicate](https://replicate.com/) (pour le modèle GPT-4o)
- Un compte [SerpAPI](https://serpapi.com/) (optionnel, pour la recherche de prix)

## 📦 Installation

1. **Cloner le dépôt :**
   ```bash
   git clone [https://github.com/votre-utilisateur/touquand-api.git](https://github.com/votre-utilisateur/touquand-api.git)
   cd touquand-api
