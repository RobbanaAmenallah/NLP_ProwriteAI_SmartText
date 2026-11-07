import pandas as pd
from datasets import load_dataset

def download_and_export():
    dataset_name = "grammarly/coedit"
    print(f"Téléchargement du dataset {dataset_name}...")
    
    # Charge le dataset complet
    dataset = load_dataset(dataset_name)
    
    # Exporte chaque split (train, validation, etc.) dans son propre fichier CSV
    for split in dataset.keys():
        output_csv = f"coedit_{split}.csv"
        print(f"Exportation du split '{split}' vers {output_csv}...")
        
        # Conversion directe en DataFrame Pandas puis en CSV
        df = dataset[split].to_pandas()
        df.to_csv(output_csv, index=False, encoding="utf-8")
        print(f"Terminé : {output_csv} ({len(df):,} lignes enregistrées)")

if __name__ == "__main__":
    download_and_export()