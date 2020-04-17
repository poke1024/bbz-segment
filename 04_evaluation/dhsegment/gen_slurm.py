#!/usr/bin/python3

from pathlib import Path

configs = []
for p in Path("/home/sc.uni-leipzig.de/bo140rasi/data/preprocessed").iterdir():
	if p.is_dir():
		for mode in ("blk", "blkx", "sep"):
			config_path = p / "dhsegment"/ mode/ "config.json"
			if config_path.exists():
				configs.append(config_path)

with open("dhs.slurm.sh", "w") as f:
	f.write("""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=dh-blk
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --partition=clara-job
#SBATCH --time=80:00:00
#SBATCH --mem-per-cpu=16384

# die fuer den Job benoetigten Module laden
module purge
module load TensorFlow/1.15.2-fosscuda-2019b-Python-3.7.4

# env
export OMPI_MCA_btl_openib_if_exclude=mlx5_bond_0

# train
""")

	for config in configs:
		f.write("python3 /home/sc.uni-leipzig.de/bo140rasi/dhs/dhSegment/train.py with %s\n" % config)

