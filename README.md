# Folia — Analyse des feuilles de tomate

Folia est une application qui permet d’analyser une photographie de feuille de tomate et de comparer les prédictions de trois modèles GoogLeNet entraînés à différentes échelles d’observation.

Le projet explore l’influence du cadrage sur la reconnaissance des maladies foliaires : une vue de la feuille entière, une portion centrale ou un gros plan peuvent conduire à des prédictions différentes.

## Fonctionnalités

- Import d’une photographie au format JPG, PNG ou BMP.
- Aperçu des trois cadrages utilisés pour l’analyse.
- Sélection d’un ou de plusieurs modèles à comparer.
- Affichage de la classe prédite et des probabilités pour chaque modèle.
- Comparaison de l’accord ou des divergences entre les modèles.
- Export des résultats au format CSV.

## Classes reconnues

L’application distingue quatre états de la feuille de tomate :

| Classe | Description |
| --- | --- |
| Feuille saine | Absence de maladie parmi les classes étudiées |
| Alternariose | Early blight |
| Mildiou | Late blight |
| Septoriose | Septoria leaf spot |

## Trois échelles d’observation

Chaque modèle analyse un cadrage différent de la même photographie.

| Modèle | Vue analysée |
| --- | --- |
| Global | Image entière |
| Intermédiaire | Recadrage central conservant 60 % de la largeur et de la hauteur |
| Local | Recadrage central conservant 35 % de la largeur et de la hauteur |

Les vues sont redimensionnées à 224 × 224 pixels avant l’analyse. Les trois modèles reposent sur l’architecture GoogLeNet et ont été entraînés sur quatre classes de tomate issues de PlantVillage.

## Parcours dans l’application

1. Importer une photo nette avec la feuille placée au centre.
2. Choisir les modèles à utiliser.
3. Cliquer sur **Analyser la feuille**.
4. Consulter les prédictions et, si souhaité, télécharger les résultats.

## Interprétation des résultats

Les probabilités indiquent les scores attribués aux quatre classes apprises. Elles ne constituent pas une mesure de fiabilité sur une photographie donnée.

Folia est un projet expérimental. Les modèles attribuent une classe à toute image sans vérifier qu’elle représente une feuille de tomate. Les résultats sur des photos de terrain peuvent différer de ceux obtenus sur les images de PlantVillage.

## Technologies

Python · Streamlit · PyTorch · Torchvision · Pillow · Pandas
