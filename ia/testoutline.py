from enum import Enum

import outlines.models as models
import outlines.text.generate as generate
import torch
from pydantic import BaseModel, constr


class Weapon(str, Enum):
    sword = "sword"
    axe = "axe"
    mace = "mace"
    spear = "spear"
    bow = "bow"
    crossbow = "crossbow"


class Armor(str, Enum):
    leather = "leather"
    chainmail = "chainmail"
    plate = "plate"


class Character(BaseModel):
    name: constr(max_length=10)
    age: int
    armor: Armor
    weapon: Weapon
    strength: int


class CandidateSentence(BaseModel):
    profile: str
    experience: str
    formation: str
    ou: str
    qui: str
    ancien_boulot: str


model = models.transformers(
    "erhwenkuo/CKIP-Llama-2-7b-chat-GTPQ", device="cuda", is_decoder=True
)

# Construct guided sequence generator
generator = generate.json(model, CandidateSentence, max_tokens=100)

# Draw a sample
rng = torch.Generator(device="cuda")
rng.manual_seed(789001)
prompt = """transfome ce text en json qui contient les champs suivants: profile, experience, formation, ou, qui, ancien_boulot"""
description = """<p><strong>#çamatchentrenous</strong></p>\n<p>✅ Un magasin en plein développement</p>\n<p>✅ Autonomie et responsabilités</p>\n<p>✅ Une société reconnue au niveau national</p>\n<p>🗣 \" Enseigne reconnu dans le secteur de la grande distribution, nous continuons de nous développer par de nouveaux défis managériaux et commerciaux ! Aujourd'hui, pour continuer notre développement, nous recherchons un(e)<strong> Chef secteur PGC H/F </strong>pour faire partie de nos équipes.\" 👨🏻‍💻</p>\n<p>
<strong>Votre potentiel permettra de:</strong></p>\n<p>👉 <strong>Animer, Encadrer, Accompagner les équipes</strong> en instaurant une véritable dynamique centrée sur la <strong>satisfaction client</strong>.</p>\n<p>👉 <strong>Assurer</strong> la gestion du personnel et le <strong>suivi RH.</strong></p>\n<p>👉 <strong>Développer l'attractivité </strong>des rayons Épicerie, Liquide et DPH grâce aux opérations commerciales.</p>\n<p>👉 <strong>Maitriser</strong> l’approvisionnement et faire respecter la réglementation en vigueur en termes de qualité et de sécurité.</p>\n<p>👉 <strong>Piloter </strong>l'ensemble du compte d'exploitation de votre entité afin de développer le CA et la rentabilité.</p>\n<p><strong>Votre envie de nous rejoindre :</strong></p>\n<p>Votre<strong> personnalité</strong> et votre <strong>talent </strong>seront les ingrédients de votre réussite : Véritable <strong>entrepreneur(se) </strong>dans l'âme, vous avez le sens des <strong>responsabilités </strong>et de l'<strong>autonomie</strong>, un esprit d'analyse et d'anticipation.</p>\n<p>Issu(e) d'une formation en commerce, vous avez une expérience significative sur un poste à responsabilités managériales.</p>\n<p>En fonction de votre expérience et de vos compétences, votre rémunération globale annuelle sera entre 40<strong>K€ à 50K€ </strong>(+ Avantages : 13ᵉ mois, ...)</p>\n<p>Poste en CDI-Statut Cadre</p>\n<p>Avec Work&You <strong>\"Provoquer LA rencontre professionnelle\"</strong> !</p>\n<p>Aurélie s'engage à vous répondre sous 48H00 !</p>",
    """
sequence = generator(prompt + " " + description, rng=rng)
print("test1", sequence)


# sequence = generator("Give me an interesting character description", rng=rng)
# print("test2", sequence)


# try:
#     parsed = Character.model_validate_json(sequence)
#     print(parsed)
# except Exception as e:
#     print("Erreur de validation JSON :", e)
