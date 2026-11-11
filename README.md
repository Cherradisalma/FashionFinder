# Classification de vêtements femme

Projet de data science : on envoie une photo, le modèle prédit la catégorie
(chemises, jupes, pantalons, pulls, robes) par transfer learning (MobileNetV2 + Keras).

## Utilisation
1. Copier vos photos dans `data/raw/<categorie>/`
2. `pip install -r requirements.txt`
3. Entraîner : `python src/train.py`
4. Évaluer : `python src/evaluate.py`
5. Lancer l'application : `streamlit run app/app.py`

(Lancer les commandes depuis la racine du projet.)
