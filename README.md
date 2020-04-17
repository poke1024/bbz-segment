# bbz-segment
This repository contains code and data for the paper at <a href="http://arxiv.org/abs/2004.07317">An Evaluation of DNN Architectures for Page Segmentation of Historical Newspapers</a>:

* `00_demo_data` gives sample data that can be used to run the script in `02_preprocessing`. Our full annotated data that was used in the paper can be found on <a href="https://www.dropbox.com/sh/4b1ub2bmmgmbprp/AAC88d8h8oZVgt-4WC5_uNloa?dl=0">Dropbox</a>.

* `01_selection` contains a random page selection script.

* `02_preprocessing` contains the full pipeline used to postprocess the ground truth (before DNN training).

* `03_training` contains the code used to train the DNN networks.

* `04_evaluation` contains various scripts for evaluating performance, as well as our raw data (as <a href="https://github.com/IDSIA/sacred">sacred</a> runs, see `04_evaluation/data`).

