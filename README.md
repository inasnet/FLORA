# Folia — Analyse des feuilles de tomate

Double-cliquez sur Lancer_interface.bat, puis ouvrez http://localhost:8501.

Installation si nécessaire :
C:\Users\HP\anaconda3\python.exe -m pip install -r requirements.txt

Lancement manuel :
C:\Users\HP\anaconda3\python.exe -m streamlit run app.py --server.address 127.0.0.1

Les trois fichiers fournis sont déjà des archives de sérialisation PyTorch : ils sont copiés sous models/*.pt sans extraction.
L’ordre des classes et le prétraitement proviennent de projet_feuilles_googlenet.ipynb : saine, alternariose, mildiou, septoriose ; recadrage central 100/60/35 %, Resize 224×224, normalisation ImageNet, transform_input=True comme dans le constructeur préentraîné utilisé à l’entraînement. Aucune augmentation aléatoire en prédiction.

Importez une photo, choisissez les modèles, cliquez sur Analyser. Les résultats et leurs probabilités peuvent être exportés en CSV. Les modèles sont chargés avec weights_only=True et vérification stricte des paramètres. Aucun téléchargement de poids ImageNet n’est nécessaire.

Le calcul utilise le CPU. Les scores softmax ne sont pas calibrés et ne remplacent pas une évaluation de fiabilité. Les modèles ne détectent pas les images hors domaine.

## Déploiement sur Streamlit Community Cloud

1. Connectez-vous sur https://share.streamlit.io avec votre compte GitHub.
2. Cliquez sur Create app et sélectionnez inasnet/FLORA.
3. Branche : main. Fichier principal : app.py.
4. Dans Advanced settings, choisissez Python 3.13, utilisé pour la validation locale.
5. Cliquez sur Deploy.

Les trois modèles sont inclus dans models/. Aucun secret ni chemin Windows n’est nécessaire au déploiement. requirements.txt fixe les versions validées localement et ajoute l’index CPU officiel PyTorch pour le serveur Linux.

Documentation : https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
