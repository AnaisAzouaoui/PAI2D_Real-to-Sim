# PAI2D Real-to-Sim — Guide d'installation et de lancement

Génération automatique de scènes pour le simulateur Genesis à partir de prompts texte ou d'images.

Ce README explique  **l'installation et le lancement** de notre logiciel. Pour les détails techniques (architecture, pipelines, modèles, choix d'implémentation), voir le rapport associé dans \documentation

---

## Prérequis

- **Python 3.11** (testé sur 3.11.2)
- **macOS ou Linux** (Windows non testé)
- **Ollama** installé localement → [https://ollama.ai](https://ollama.ai)
- Une **clé API OpenAI** valide
- GPU (le backend CPU de Genesis suffit, mais c'est plus lent)

---

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/czouloumian/PAI2D_Real-to-Sim.git
cd PAI2D_Real-to-Sim
```

### 2. Créer l'environnement virtuel

```bash
python3.11 -m venv venv_new
source venv_new/bin/activate
```

### 3. Installer les dépendances Python

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

L'installation peut prendre plusieurs minutes (torch, transformers, PyQt6, Genesis sont volumineux).

### 4. Configurer la clé OpenAI

Créer un fichier `.env` à la racine du dépôt :

```
OPENAI_API_KEY=sk-...
```

Remplacer `sk-...` par votre clé OpenAI. Sans cette clé, le pipeline par défaut (qui utilise GPT-4o pour la validation sémantique et le mode image) ne fonctionnera pas.

### 5. Installer Ollama et le modèle phi3-scene (pour le pipeline V2)

Le pipeline V2 (par défaut) repose sur un modèle Phi-3 finetuné servi localement par Ollama.

**a. Installer Ollama**

Suivre les instructions sur [https://ollama.ai](https://ollama.ai) (binaire macOS, ou `curl -fsSL https://ollama.ai/install.sh | sh` sur Linux).

Une fois installé, lancer le serveur Ollama (souvent automatique au démarrage) :

```bash
ollama serve
```

**b. Récupérer le fichier GGUF du modèle finetuné**

Le fichier `phi3_finetuned_gguf.gguf` (modèle finetuné, format Q4_K_M) ainsi que son `Modelfile` doivent etre récuperer à partir du lien suivant : https://drive.google.com/drive/folders/1rwmoFAQamb7Hd8Rs39S7H2NZFriomx56?usp=share_link et doivent être placés dans un dossier accessible.

**c. Créer le modèle dans Ollama**

Depuis le dossier contenant le `Modelfile` :

```bash
ollama create phi3-scene -f Modelfile
```

**d. Vérifier l'installation**

```bash
ollama run phi3-scene "test"
```

Si le modèle répond, c'est prêt (ce modèle étant entrainé sur une tache spécifique, "test" lui fera probablement rendre du charabia, vous serez au moins certains qu'il est installé :D)

---

## Lancement

Une fois l'environnement activé (`source venv_new/bin/activate`) :

```bash
python src/app.py
```

L'interface graphique PyQt6 s'ouvre. Vous pouvez alors :

- Saisir un prompt en langage naturel (français ou anglais)
- Joindre une image via le bouton trombone (pour les pipelines V3 / V3.1)
- Sélectionner la version de pipeline dans le menu déroulant
- Lancer la simulation Genesis ou exporter la scène en JSON

---

## Problèmes fréquents

**`ModuleNotFoundError`** -> l'environnement virtuel n'est pas activé. Refaire `source venv_new/bin/activate`.

**`ConnectionError: Impossible de joindre Ollama`** -> le serveur Ollama n'est pas lancé. Faire `ollama serve` dans un autre terminal.

**`OPENAI_API_KEY` invalide** -> vérifier le contenu du fichier `.env` à la racine et la validité de la clé sur [platform.openai.com](https://platform.openai.com).

**Erreur de chargement d'un modèle Hugging Face** -> la première utilisation des pipelines V3.1, V1.1.1 ou des fonctions de profondeur télécharge des modèles (GroundingDINO, Depth Anything V2, embeddings multilingues). Vérifier la connexion internet et l'espace disque (environ 5 Go au total).


Auteurs : Anais Azouaoui, Caroline Zouloumian. 
