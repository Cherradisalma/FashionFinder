import sys, os, random
sys.path.append("src")
import streamlit as st
from PIL import Image
from predict import predict_all

st.title("Détection de vêtements femme")

seuil = st.sidebar.slider("Seuil de confiance", 0.30, 0.95, 0.60, 0.05)
st.sidebar.write("Sous ce seuil, le vêtement est considéré comme non trouvé.")

file = st.file_uploader("Choisissez une photo", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file)
    st.image(img, use_container_width=True)

    results = predict_all(img)
    label, conf = results[0]
    rejete = (label == "autres") or (conf < seuil)

    if rejete:
        st.error("Non trouvé : ce vêtement n'existe pas dans les catégories du modèle.")
        st.caption(f"Meilleure proposition : {label} ({conf:.1%})")
    else:
        st.success(f"Catégorie : **{label}** ({conf:.1%})")

    st.subheader("Ressemblance avec chaque catégorie")
    for name, p in results:
        st.write(f"**{name}** : {p:.1%}")
        st.progress(float(p))

    if not rejete:
        st.subheader(f"Exemples de la catégorie « {label} »")
        folder = os.path.join("data", "raw", label)
        examples = random.sample(os.listdir(folder), 3)
        st.image([os.path.join(folder, f) for f in examples], width=150)