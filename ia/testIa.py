import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print(torch.cuda.is_available())
print(torch.cuda.memory_allocated())
# Spécifiez le nom du modèle que vous souhaitez utiliser
model_name = "amazon/LightGPT"

# Initialisez le tokenizer et le modèle
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, return_dict=True)

# Texte d'entrée pour la génération
input_text = "Bonjour, comment ça va ?"

# Tokenisation de l'entrée
input_ids = tokenizer.encode(input_text, return_tensors="pt")
model.to("cuda")
input_ids = input_ids.to("cuda")
# Génération de texte avec le modèle
output = model.generate(
    input_ids,
    max_length=50,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    top_k=50,
)

# Décodage des résultats générés en texte
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Affichage du texte généré
print("Texte généré :")
print(generated_text)
