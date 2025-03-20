# 🤖 Assistant IA pour la Recherche d'Emploi

## 📝 Description du Projet
Ce projet est un assistant intelligent qui aide les chercheurs d'emploi à optimiser leur recherche et à améliorer leurs candidatures. L'application utilise l'Intelligence Artificielle pour analyser les CV, les descriptions de postes et fournir des recommandations personnalisées.

## 🏗️ Architecture du Projet

### 1. Collecte de Données
- **Sources de données**: Sites de recherche d'emploi (scraping via Selenium)
- **Stockage**: Base de données vectorielle Qdrant (cloud)
- **Technologies**: Python, Selenium, Qdrant

### 2. Analyse et Traitement
- **Analyse de CV**: Extraction et analyse des informations clés
- **Matching de Postes**: Comparaison entre CV et descriptions de postes
- **Optimisation**: Suggestions d'amélioration basées sur l'IA

### 3. Interface Utilisateur
- **Framework**: Streamlit
- **Langues**: Support multilingue (Français, Anglais, Chinois)
- **Design**: Interface intuitive et responsive

## 🚀 Fonctionnalités Principales

### Analyse de CV
- 📄 Analyse détaillée du contenu du CV
- 🎯 Identification des points forts et des axes d'amélioration
- 🔍 Optimisation pour les systèmes ATS

### Recherche d'Emploi
- 🔎 Recherche intelligente de postes correspondants
- 📊 Score de compatibilité entre CV et postes
- 🎯 Suggestions de postes pertinents

### Optimisation de Candidature
- ✍️ Suggestions d'amélioration du CV
- 🎯 Personnalisation selon le poste cible
- 📈 Optimisation pour les systèmes de recrutement

### Recherche d'Entreprise
- 🏢 Analyse des entreprises cibles
- 📊 Informations sur la culture d'entreprise
- 💡 Préparation aux entretiens

## 🛠️ Technologies Utilisées
- **Backend**: Python, CrewAI, Pydantic
- **Base de données**: Qdrant (vectorielle)
- **IA**: Vertex AI, Gemini
- **Web Scraping**: Selenium
- **Frontend**: Streamlit
- **Outils**: FileReadTool, MDXSearchTool, SerperDevTool

## 📊 Structure des Données
- **CV**: Stockage et analyse des informations du candidat
- **Postes**: Base de données des offres d'emploi scrapées
- **Entreprises**: Informations sur les entreprises cibles
- **Analyses**: Résultats des analyses et recommandations

## 🔄 Processus de Fonctionnement
1. Upload du CV par l'utilisateur
2. Analyse du CV par l'IA
3. Recherche de postes correspondants dans la base Qdrant
4. Génération de recommandations personnalisées
5. Création d'un rapport détaillé

## 🎯 Points d'Amélioration Futurs
- Optimisation des performances du scraping
- Amélioration de la précision des analyses
- Ajout de nouvelles sources de données
- Expansion des fonctionnalités d'analyse
- Optimisation de l'utilisation des ressources IA

## 🚀 Installation et Utilisation
1. Cloner le repository
2. Installer les dépendances
   ```bash
   pip install -r requirements.txt
   ```
3. Configurer les variables d'environnement
   - Copiez le fichier `.env.example` en `.env`
   ```bash
   cp .env.example .env
   ```
   - Remplissez le fichier `.env` avec vos propres clés API
   - Vous aurez besoin de créer des comptes sur:
     - [Qdrant Cloud](https://cloud.qdrant.io/) pour la base de données vectorielle
     - [OpenAI](https://platform.openai.com/) pour GPT
     - [Google AI Studio](https://ai.google.dev/) pour Gemini
     - [Anthropic](https://console.anthropic.com/) pour Claude
     - [SerperDev](https://serper.dev/) pour les recherches
     - [Hugging Face](https://huggingface.co/) pour les modèles
4. Lancer l'application
   ```bash
   streamlit run src/gojob/crewai_st.py
   ```

## 📝 Contribution
Les contributions sont les bienvenues ! N'hésitez pas à :
- Signaler des bugs
- Proposer des améliorations
- Contribuer au code
- Améliorer la documentation
