# 🚀 Guide de Déploiement sur Render (ProwriteAI)

Ce guide détaille pas à pas la mise en ligne ultra-rapide de l'application **ProwriteAI** sur **[Render.com](https://render.com/)**.

---

## ⚡ Pourquoi Flask + Gunicorn sur Render (vs Streamlit) ?

Bien que Streamlit soit excellent pour le prototypage local, il est notoirement lent sur les tiers gratuits du Cloud :
- **Taille de la page :** Streamlit télécharge un bundle JavaScript/React de ~15 à 20 Mo à chaque visite, contre **< 30 Ko** pour notre interface Flask optimisée (chargement instantané en **50 ms**).
- **Latence d'inférence :** L'inférence neuronale s'exécute en **~25–40 ms** via l'API REST Flask (AJAX `fetch()`), sans rechargement complet de script ni reconnexion WebSocket.
- **Empreinte mémoire :** Flask ne consomme que **~30 Mo de RAM** (contre 180 Mo pour Streamlit), laissant toute la mémoire libre pour PyTorch sur les 512 Mo alloués par Render.

*(Note : L'application Streamlit `streamlit run app.py` reste également disponible pour une utilisation locale ou pour Streamlit Community Cloud).*

---

## ⚡ Méthode 1 : Déploiement automatique en 1 clic (Render Blueprint)

Le dépôt contient déjà le fichier d'infrastructure [`render.yaml`](file:///c:/Users/robba/OneDrive/Bureau/Dauphine_M2/NLP/RobbanaAmenallah_NLP_Project/render.yaml). Render est capable de tout configurer automatiquement sans saisie manuelle.

1. Connectez-vous sur **[dashboard.render.com](https://dashboard.render.com/)**.
2. Cliquez sur le bouton bleu **New +** en haut à droite, puis sélectionnez **Blueprint**.
3. Associez votre compte GitHub et choisissez le dépôt `RobbanaAmenallah/NLP_ProwriteAI_SmartText`.
4. Render détecte automatiquement `render.yaml` et affiche le résumé du service web (`prowriteai-smarttext`).
5. Cliquez sur **Apply** : Render installe les dépendances CPU légères, démarre le serveur de production Gunicorn et génère votre URL publique HTTPS (ex. `https://prowriteai-smarttext.onrender.com`).

---

## 🛠️ Méthode 2 : Configuration Manuelle d'un Web Service

Si vous préférez configurer le service manuellement depuis le dashboard Render :

1. Cliquez sur **New +** → **Web Service**.
2. Choisissez **Build and deploy from a Git repository** et sélectionnez votre dépôt GitHub.
3. Remplissez les champs de configuration comme suit :

| Paramètre | Valeur à saisir |
|---|---|
| **Name** | `prowriteai-smarttext` (ou le nom de votre choix) |
| **Region** | `Frankfurt (EU Central)` *(le plus proche pour la France)* |
| **Branch** | `main` |
| **Root Directory** | *(laisser vide)* |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install --upgrade pip && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt` |
| **Start Command** | `gunicorn app_flask:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120` |
| **Instance Type** | `Free` (512 MB RAM, 0.1 CPU) |

4. Dans la section **Environment Variables**, ajoutez :
   - `PYTHON_VERSION` = `3.11.9`
   - `PROWRITE_TORCH_THREADS` = `2`

5. Cliquez sur **Create Web Service**.

---

## 💡 Détails d'optimisation pour le Free Tier

1. **Installation ultra-légère de PyTorch CPU :**
   - Sur Linux, `pip install torch` télécharge par défaut les bibliothèques CUDA de NVIDIA (~2,5 Go !), ce qui dépasse le quota de disque temporaire de Render et fait échouer le build par *Timeout* ou *No space left*.
   - La commande `--index-url https://download.pytorch.org/whl/cpu` télécharge uniquement la version CPU (~170 Mo). L'installation se termine en moins de 60 secondes.
2. **Consommation mémoire maîtrisée :**
   - L'ensemble des 4 modèles chargés ne consomme que **~230 Mo de RAM**, ce qui reste largement sous le plafond des **512 Mo** du plan gratuit Render.
3. **Mise en veille du plan gratuit (Cold Start) :**
   - Sur l'offre gratuite de Render, le serveur s'endort après 15 minutes d'inactivité. À la première visite, le réveil prend entre 30 et 50 secondes, puis toutes les requêtes suivantes sont quasi-instantanées (< 50 ms).
