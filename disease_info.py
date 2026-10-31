"""
Disease information shown AFTER the classifier predicts a class.

Only well-known, general facts are included. No pesticide / medicine names or
dosages are given on purpose - this is an academic project.
If the predicted class is not in the table, general management advice is shown.
"""
import re

DISCLAIMER = ("This is an academic project and the result should be verified by an "
              "agricultural expert before treatment.")

GENERAL_MANAGEMENT = [
    "Remove and destroy heavily affected leaves; do not compost them.",
    "Improve air circulation (spacing, pruning) and avoid wetting leaves when watering.",
    "Keep tools clean and avoid working among wet plants to limit spread.",
    "Choose resistant varieties and keep the area free of fallen plant debris.",
    "Ask a local agricultural expert / extension office which treatment, if any, is suitable.",
]

# Each entry: keywords that must ALL appear in the normalised label.
_DB = [
    {
        "match": ["black", "rot"],
        "title": "Black Rot",
        "symptoms": "Small brown circular spots on leaves that enlarge, with a dark border and tiny black dots (fruiting bodies) in the centre; fruit may shrivel into hard black 'mummies'.",
        "cause": "Fungus Guignardia bidwellii (grape black rot); favoured by warm, wet weather.",
        "management": ["Remove mummified fruit and infected leaves/canes.",
                       "Prune for open canopy so foliage dries quickly.",
                       "Follow local expert guidance on protective spray programmes."],
    },
    {
        "match": ["esca"],
        "title": "Esca (Black Measles)",
        "symptoms": "Yellow or brown 'tiger-stripe' patterns between leaf veins, dark spots on berries, and wood decay in older vines.",
        "cause": "A complex of wood-inhabiting fungi that enter through pruning wounds.",
        "management": ["Prune in dry weather and protect large pruning wounds.",
                       "Remove and destroy dead or badly affected vines/wood.",
                       "Consult an expert - there is no simple leaf-only cure."],
    },
    {
        "match": ["black", "measles"],
        "title": "Esca (Black Measles)",
        "symptoms": "Yellow or brown 'tiger-stripe' patterns between leaf veins, dark spots on berries, and wood decay in older vines.",
        "cause": "A complex of wood-inhabiting fungi that enter through pruning wounds.",
        "management": ["Prune in dry weather and protect large pruning wounds.",
                       "Remove and destroy dead or badly affected vines/wood."],
    },
    {
        "match": ["grape", "blight"],
        "title": "Grape Leaf Blight",
        "symptoms": "Irregular reddish-brown to dark patches on leaves that may merge and cause early leaf drop.",
        "cause": "Fungus Pseudocercospora vitis (Isariopsis leaf spot) in humid conditions.",
        "management": ["Remove infected leaves and fallen debris.",
                       "Improve canopy airflow and avoid overhead irrigation."],
    },
    {
        "match": ["powdery", "mildew"],
        "title": "Powdery Mildew",
        "symptoms": "White to grey powdery patches on the leaf surface, which may curl, yellow or dry out.",
        "cause": "Powdery mildew fungi (order Erysiphales); thrives in warm days, cool humid nights and crowded foliage.",
        "management": ["Prune for airflow and avoid excess nitrogen fertiliser.",
                       "Remove badly affected leaves.",
                       "Ask an expert about suitable preventive treatment for the crop."],
    },
    {
        "match": ["bacterial", "spot"],
        "title": "Bacterial Spot",
        "symptoms": "Small dark, water-soaked angular spots that may turn purple-brown and drop out, leaving a 'shot-hole' look.",
        "cause": "Bacteria (e.g. Xanthomonas species); spread by rain splash and wet foliage.",
        "management": ["Avoid overhead watering and working in wet foliage.",
                       "Remove infected leaves; use clean planting material.",
                       "Consult an expert for crop-specific control options."],
    },
    {
        "match": ["leaf", "spot"],
        "title": "Leaf Spot",
        "symptoms": "Round to irregular brown or dark spots on leaves, sometimes with a yellow halo.",
        "cause": "Usually fungal or bacterial pathogens; exact cause depends on the plant and dataset class.",
        "management": GENERAL_MANAGEMENT[:4],
    },
    {
        "match": ["rust"],
        "title": "Rust",
        "symptoms": "Orange, yellow or brown pustules (often on the underside of leaves) and yellowing of the leaf above.",
        "cause": "Rust fungi (order Pucciniales).",
        "management": GENERAL_MANAGEMENT[:4],
    },
    {
        "match": ["scab"],
        "title": "Scab",
        "symptoms": "Olive-green to dark, velvety or scabby lesions on leaves and fruit.",
        "cause": "Fungal infection favoured by cool, wet spring weather.",
        "management": GENERAL_MANAGEMENT[:4],
    },
]


def pretty_label(label: str) -> str:
    """'Grape___Black_rot' -> 'Grape - Black rot'; 'Grape_Black_Rot' -> 'Grape Black Rot'."""
    s = label.strip()
    s = re.sub(r"_{2,}", " - ", s)
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return " ".join(w[:1].upper() + w[1:] if w.islower() else w for w in s.split(" "))


def get_disease_info(label: str) -> dict:
    """Return display info for a predicted class label."""
    norm = re.sub(r"[^a-z]+", " ", label.lower())
    name = pretty_label(label)

    if "healthy" in norm:
        return {
            "name": "Healthy Leaf", "known": True,
            "symptoms": "No disease symptoms were recognised by the classifier.",
            "cause": "Not applicable.",
            "management": ["Keep monitoring the plant regularly.",
                           "Maintain good watering, spacing and sanitation to prevent disease."],
            "disclaimer": DISCLAIMER,
        }

    for entry in _DB:
        if all(k in norm.split() or k in norm for k in entry["match"]):
            return {"name": name, "known": True, "symptoms": entry["symptoms"],
                    "cause": entry["cause"], "management": entry["management"],
                    "disclaimer": DISCLAIMER}

    return {
        "name": name, "known": False,
        "symptoms": "Detailed symptom information for this class is not stored in the project. "
                    "Compare the highlighted region with reference images of this disease.",
        "cause": "Not available for this class in the project database.",
        "management": GENERAL_MANAGEMENT,
        "disclaimer": DISCLAIMER,
    }
