#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=blkx
#SBATCH --gres=gpu:v100:4
#SBATCH --partition=clara-job
#SBATCH --time=30:00:00
#SBATCH --mem-per-cpu=16384

# die fuer den Job benoetigten Module laden
module purge
module load TensorFlow/1.15.2-fosscuda-2019b-Python-3.7.4

#pip3 install --user segmentation_models sacred albumentations sklearn keras_lookahead tqdm

# ein paar Umgebungsvariablen setzen
export BBZ_SEGMENT_RUNS_PATH=/home/sc.uni-leipzig.de/bo140rasi/runs-blkx
export BBZ_SEGMENT_DATA_PATH=/home/sc.uni-leipzig.de/bo140rasi/data/training

export OMPI_MCA_btl_openib_if_exclude=mlx5_bond_0

# Das Programm ausfuehren
#python train.py with "data_name=layout_896x1280" "data_path=$BBZ_SEGMENT_DATA_PATH" "ngpus=4" "batchsize=24"
python grid.py blkx.json
