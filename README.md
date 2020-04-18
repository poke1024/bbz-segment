# bbz-segment
This repository contains code and data for the paper <a href="http://arxiv.org/abs/2004.07317">An Evaluation of DNN Architectures for Page Segmentation of Historical Newspapers</a>:

* `00_demo_data` gives sample data that can be used to run the script in `02_preprocessing`. Our full annotated data that was used in the paper can be found on <a href="https://www.dropbox.com/sh/4b1ub2bmmgmbprp/AAC88d8h8oZVgt-4WC5_uNloa?dl=0">Dropbox</a>.

* `01_selection` contains a random page selection script.

* `02_preprocessing` contains the full pipeline used to postprocess the ground truth (before DNN training).

* `03_training` contains the code used to train the DNN networks. Note that `train.py` contains AdamW optimizer code copied from https://github.com/OverLordGoldDragon/keras-adamw.

* `04_evaluation` contains various scripts for evaluating performance, as well as our raw data (as <a href="https://github.com/IDSIA/sacred">sacred</a> runs, see `04_evaluation/data`).

* `05_prediction` gives scripts for running our final models for prediction (see graphics below for the demo result). To run it yourself on on this or other document images, first download the models from <a href="https://www.dropbox.com/sh/7tph1tzscw3cb8r/AAA9WxhqoKJu9jLfVU5GqgkFa?dl=0">Dropbox</a> and move them to `05_prediction/data/models`. Then run `05_prediction/src/main.py` to predict the files in `05_prediction/data/pages`. Note that you need to have numpy, tensorflow and <a href="https://github.com/qubvel/segmentation_models">segmentation_models</a> installed.

## Demo Page

![Page](05_prediction/data/pages/2436020X_1925-02-27_70_98_008.jpg)

## Predicted SEP labels

![Prediction for sep](05_prediction/demo/2436020X_1925-02-27_70_98_008.sep.png)
**Legend**. Red: Background, Orange: Horizontal Separators, Green: Vertical Separators, Blue: Table Column Separators.

## Predicted BLKX labels

![Prediction for blkx](05_prediction/demo/2436020X_1925-02-27_70_98_008.blkx.png)
**Legend**. Red: Background, Blue: Text Region, Orange: Table Region, Green: Illustrations/Borders.
