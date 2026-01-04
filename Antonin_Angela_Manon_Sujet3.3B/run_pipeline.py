"""
Script principal pour exécuter tout le pipeline du projet.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    print(f"\n{'='*50}")
    print(f"🚀 {description}")
    print(f"{'='*50}")
    try:
        subprocess.run(command, check=True, shell=True)
        print(f"✅ {description} terminé avec succès.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de {description}: {e}")
        sys.exit(1)

def main():
    base_dir = Path(__file__).parent
    src_dir = base_dir / "src"
    
    # 1. Téléchargement des données
    run_command(f'"{sys.executable}" "{src_dir / "data" / "download_data.py"}"', "Téléchargement et préparation des données")
    
    # 2. Entraînement Baseline
    run_command(f'"{sys.executable}" "{src_dir / "training" / "train_baseline.py"}"', "Entraînement de la Baseline")
    
    # 3. Entraînement CamemBERT
    # On utilise python directement. Assurez-vous que les dépendances sont installées.
    run_command(f'"{sys.executable}" "{src_dir / "training" / "train.py"}"', "Entraînement de CamemBERT Multi-tâches")
    
    print("\n🎉 Pipeline terminé ! Vous pouvez maintenant lancer l'application de démo :")
    print(f"streamlit run src/app/app.py")

if __name__ == "__main__":
    main()
