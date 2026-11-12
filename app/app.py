import sys, os, random
sys.path.append("src")
import streamlit as st
from PIL import Image
from predict import predict_all

st.set_page_config(page_title="FashionFinder", page_icon="👗", layout="wide")

EMOJIS = {"chemises": "👔", "jupes": "🎀", "pantalons": "👖",
          "pulls": "🧶", "robes": "👗", "autres": "❓"}

st.markdown("""
<style>
.hero {background: linear-gradient(135deg, #ff6b9d, #c44dff); padding: 2rem;
       border-radius: 18px; color: white; text-align: center; margin-bottom: 1.5rem;}
.hero h1 {margin: 0; font-size: 2.4rem; color: white;}
.hero p {margin: .4rem 0 0 0; opacity: .9;}
.result {padding: 1rem 1.2rem; border-radius: 14px; font-size: 1.2rem;
         font-weight: 600; margin-bottom: 1rem;}
.ok {background: #e8f8ee; color: #1b7a3d; border-left: 6px solid #2ecc71;}
.ko {background: #fdecea; color: #a93226; border-left: 6px solid #e74c3c;}
.bar-label {display: flex; justify-content: space-between; margin-top: .7rem; font-size: .95rem;}
.bar {background: #e6e6e6; border-radius: 10px; height: 12px; overflow: hidden;}
.bar-fill {height: 100%; border-radius: 10px; background: linear-gradient(90deg, #ff6b9d, #c44dff);}
</style>
""", unsafe_allow_html=True)

# ----- Barre latérale -----
# st.sidebar.title("⚙️ Réglages")
# seuil = st.sidebar.slider("Seuil de confiance", 0.30, 0.95, 0.60, 0.05)
# st.sidebar.caption("Sous ce seuil, le vêtement est considéré comme non trouvé.")
st.sidebar.markdown("**Catégories reconnues**")
for nom in ["chemises", "jupes", "pantalons", "pulls", "robes"]:
    st.sidebar.write(f"{EMOJIS[nom]} {nom}")

# ----- En-tête -----
st.markdown("""
<div class="hero">
<h1>👗 FashionFinder</h1>
<p>Envoyez une photo, le modèle reconnaît la catégorie du vêtement.</p>
</div>
""", unsafe_allow_html=True)

file = st.file_uploader("Choisissez une photo", type=["jpg", "jpeg", "png"])

if not file:
    st.info("Envoyez une photo de chemise, jupe, pantalon, pull ou robe pour commencer.")
else:
    img = Image.open(file)
    results = predict_all(img)
    label, conf = results[0]
    rejete = (label == "autres") or (conf < seuil)

    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.image(img, use_container_width=True)

    with col2:
        if rejete:
            st.markdown('<div class="result ko">❌ Non trouvé : ce vêtement n\'existe pas dans les catégories du modèle.</div>', unsafe_allow_html=True)
            st.caption(f"Meilleure proposition : {label} ({conf:.1%})")
        else:
            st.markdown(f'<div class="result ok">{EMOJIS.get(label, "")} Catégorie : {label} ({conf:.1%})</div>', unsafe_allow_html=True)

        st.subheader("Ressemblance avec chaque catégorie")
        for name, p in results:
            st.markdown(
                f'<div class="bar-label"><span>{EMOJIS.get(name, "")} {name}</span><span>{p:.1%}</span></div>'
                f'<div class="bar"><div class="bar-fill" style="width:{p*100:.1f}%"></div></div>',
                unsafe_allow_html=True)

    if not rejete:
        st.divider()
        st.subheader(f"Exemples de la catégorie « {label} »")
        folder = os.path.join("data", "raw", label)
        fichiers = os.listdir(folder)
        exemples = random.sample(fichiers, min(3, len(fichiers)))
        cols = st.columns(3)
        for c, f in zip(cols, exemples):
            c.image(os.path.join(folder, f), use_container_width=True)

# st.divider()
# st.caption("Projet de data science · MobileNetV2 + Keras · Streamlit")