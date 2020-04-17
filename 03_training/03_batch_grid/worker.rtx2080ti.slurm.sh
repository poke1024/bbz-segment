#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=worker
#SBATCH --gres=gpu:rtx2080ti:4
#SBATCH --partition=clara-job
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=16384

# die fuer den Job benoetigten Module laden
module purge
module load TensorFlow/1.15.2-fosscuda-2019b-Python-3.7.4

#pip3 install --user segmentation_models sacred albumentations sklearn keras_lookahead tqdm

# ein paar Umgebungsvariablen setzen
export BBZ_SEGMENT_RUNS_PATH=/home/sc.uni-leipzig.de/bo140rasi/runs
export BBZ_SEGMENT_DATA_PATH=/home/sc.uni-leipzig.de/bo140rasi/data/training

export OMPI_MCA_btl_openib_if_exclude=mlx5_bond_0

# Das Programm ausfuehren
python gridmate.py work rtx2080ti
