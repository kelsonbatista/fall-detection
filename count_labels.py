import os
from collections import defaultdict

# Caminho para os diretórios das anotações
dataset_path = "dataset/fall"

# Classes do YOLO
classes = {0: "no_fall", 1: "fall", 2: "attention"}  # Ajuste se necessário

# Função para contar os rótulos
def count_labels(label_dir):
    label_count = defaultdict(int)
    
    # Percorre todos os diretórios dentro do label_dir
    for root, _, files in os.walk(label_dir):
        for file in files:
            if file.endswith(".txt"):  # Apenas arquivos de anotação YOLO
                with open(os.path.join(root, file), "r") as f:
                    for line in f:
                        class_id = int(line.split()[0])  # Pega o primeiro número da linha (ID da classe)
                        label_count[classes.get(class_id, "unknown")] += 1
    
    return label_count

# Contar labels no dataset
for subset in ["train", "val", "test"]:
    print(f"\n### Contagem em {subset.upper()} ###")
    fall_counts = count_labels(os.path.join(dataset_path, subset, "labels"))
    # fall_counts = count_labels(os.path.join(dataset_path, "fall", subset, "labels"))
    #no_fall_counts = count_labels(os.path.join(dataset_path, "no_fall", subset, "labels"))

    # total_counts = {label: fall_counts.get(label, 0) + no_fall_counts.get(label, 0) for label in classes.values()}
    total_counts = {label: fall_counts.get(label, 0) for label in classes.values()}
    
    for label, count in total_counts.items():
        print(f"{label}: {count} frames")

print("\nContagem finalizada.")
