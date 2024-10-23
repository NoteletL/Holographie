#!/bin/bash
#SBATCH --job-name=128SDiff   # Nom du travail
#SBATCH --output=Logs/128Soutput.log              # Fichier de sortie
#SBATCH --error=Logs/128Serror.log                # Fichier d'erreur
#SBATCH --ntasks=1                       # Nombre de tâches
#SBATCH --time=20:00:00                  # Durée maximale
#SBATCH --gres=gpu:rtx6000:1                   # Nombre de GPU
#SBATCH --mem=10G                        # Mémoire

# Charger les modules nécessairesx
module load cuda/12.4

# Exécuter le script Python
python 512QuickDenoising.py
