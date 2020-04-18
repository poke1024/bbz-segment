import os
import sys
import predict

from pathlib import Path

script_dir = Path(os.path.abspath(__file__)).parent
data_dir = script_dir.parent / "data"

models_path = data_dir / "models"
if not models_path.exists():
	print("please download models to %s." % models_path)
	sys.exit(1)

loaded = predict.load([
	(predict.NetPredictor, "v3/sep/1"),
	(predict.NetPredictor, "v3/sep/2"),
	(predict.NetPredictor, "v3/sep/4"),
	(predict.NetPredictor, "v3/blkx/2"),
	(predict.NetPredictor, "v3/blkx/4"),
	(predict.NetPredictor, "v3/blkx/5"),
	], models_path=models_path)

sep_predictor = predict.VotingPredictor(
	loaded["v3/sep/1"],
	loaded["v3/sep/2"],
	loaded["v3/sep/4"]
)

blkx_predictor = predict.VotingPredictor(
	loaded["v3/blkx/2"],
	loaded["v3/blkx/4"],
	loaded["v3/blkx/5"]
)

for page_path in (data_dir / "pages").iterdir():
	if page_path.suffix == ".jpg":
		page = predict.Page(page_path)
		sep_predictor(page).save(data_dir / (page_path.stem + ".sep.png"))
		blkx_predictor(page).save(data_dir / (page_path.stem + ".blkx.png"))
