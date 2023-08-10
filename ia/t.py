from transformers import AutoModel, AutoTokenizer


def main():
    # Charger le tokenizer et le modèle pré-entraîné
    model_name = "bert-base-uncased"  # Remplacez par le nom du modèle que vous souhaitez utiliser
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Entrée texte pour tester le modèle
    input_text = "Hello, how are you?"

    # Prétraiter le texte en token pour l'entrée du modèle
    inputs = tokenizer(input_text, return_tensors="pt")

    # Passez l'entrée tokenisée au modèle pour obtenir les embeddings
    outputs = model(**inputs)

    # Obtenez les embeddings du token "[CLS]", qui est le token spécial d'introduction du modèle
    cls_embedding = outputs.last_hidden_state[:, 0, :]

    # Faites ce que vous voulez avec les embeddings, par exemple, affichez-les
    print("Embeddings:", cls_embedding)


if __name__ == "__main__":
    main()
