BAN6440 — ANN Optimization for Customer Segmentation
This repository contains the Module 6 assignment for BAN6440 Big Data Analytics. It compares an untuned artificial neural network with optimized ANN variants using SGD with Nesterov momentum, RMSprop, and Adam on the Teleconnect customer-churn dataset.
Repository: https://github.com/willykosgeik/ban6440-ann-optimization
Project objective
The project evaluates whether optimizer selection and a controlled training pipeline improve customer-churn detection for a telecoms retention use case. The analysis focuses on the trade-off between overall accuracy and the ability to identify customers in the churn class.
The project includes:
•	A reproducible preprocessing pipeline.
•	An untuned ANN baseline.
•	Optimized ANN models using SGD, RMSprop, and Adam.
•	Validation-based early stopping and threshold selection.
•	Final evaluation on an untouched test set.
•	Confusion matrices and training-history plots.
•	Permutation-based feature importance.
•	Automated tests using pytest.
•	A run manifest containing reproducibility metadata.
Dataset
Place the supplied dataset in the project root with the following filename:
teleconnect.csv

The dataset contains 7,043 customer records and 21 raw columns. The target column is Churn, where Yes represents churn and No represents non-churn. The identifier column customerID is removed before modelling.
The preprocessing pipeline:
•	Converts TotalCharges to numeric values.
•	Handles the 11 blank TotalCharges entries.
•	Encodes the target as 0 and 1.
•	One-hot encodes categorical predictors.
•	Removes constant columns when present.
•	Fits StandardScaler on training data only.
•	Applies stratified train, validation, and test splits.
Environment
Recommended environment:
•	Python 3.11.
•	TensorFlow compatible with the local CPU environment.
•	Windows, macOS, or Linux.
Install the required packages:
python -m pip install --upgrade pip
python -m pip install tensorflow pandas numpy scikit-learn matplotlib seaborn pytest

If a requirements.txt file is included in the repository, install from it instead:
python -m pip install -r requirements.txt

Project files
The main files are:
ann_optimization_corrected.py
    Reproducible training and evaluation pipeline.

test_ann_optimization_corrected.py
    Automated tests for preprocessing, models, metrics, and thresholds.

teleconnect.csv
    Input dataset. Do not commit private or restricted customer data.

output_final/
    Generated plots, CSV files, and run manifest.

Run the tests
Activate the virtual environment if one is being used:
.\.venv\Scripts\activate

Run the test suite:
python -m pytest -v test_ann_optimization_corrected.py

A successful run should report all tests as passed. The current verified run completed with:
12 passed in 8.01s

Run the analysis
Run the corrected pipeline from the project root:
python ann_optimization_corrected.py

The script uses random seed 42 and creates the output_final directory if it does not already exist.
To reduce TensorFlow informational messages on Windows, the following optional commands may be used:
$env:TF_ENABLE_ONEDNN_OPTS="0"
$env:TF_CPP_MIN_LOG_LEVEL="2"
python ann_optimization_corrected.py

These environment variables affect logging and numerical execution settings; they do not replace the reproducibility metadata saved by the script.
Evaluation protocol
The corrected pipeline separates the data into training, validation, and test sets. The validation set is used for:
•	Early stopping.
•	Learning-rate monitoring.
•	Threshold selection.
•	Optimizer selection by validation AUC-ROC.
After the threshold is selected, it is locked. The test set is then used once for final evaluation. The test set is not used to choose the optimizer or threshold.
The tested candidate thresholds are:
0.30, 0.40, 0.50, 0.60, 0.70

The threshold is selected by the configured validation objective. The report must state the exact objective used by the final code.
Model configurations
Baseline
•	Dense layers: 16 and 8 neurons.
•	ReLU hidden activations.
•	Sigmoid output.
•	SGD optimizer.
•	Learning rate: 0.01.
•	No class weighting, batch normalization, or dropout.
Optimized variants
•	Dense layers: 32 and 16 neurons.
•	He initialization.
•	Batch normalization.
•	ReLU activations.
•	Dropout rate: 0.2.
•	Sigmoid output.
•	Binary cross-entropy loss.
•	Batch size: 64.
•	Balanced class weights.
•	Early stopping on validation AUC.
•	ReduceLROnPlateau callback.
Optimizer-specific settings:
•	SGD: learning rate 0.01, momentum 0.9, Nesterov enabled.
•	RMSprop: learning rate 0.001.
•	Adam: learning rate 0.001.
Generated outputs
The analysis generates:
output_final/model_comparison.csv
output_final/run_manifest.json
output_final/cm_baseline.png
output_final/cm_sgd.png
output_final/cm_rmsprop.png
output_final/cm_adam.png
output_final/training_baseline.png
output_final/training_sgd.png
output_final/training_rmsprop.png
output_final/training_adam.png
output_final/threshold_analysis.png
output_final/threshold_sgd.csv
output_final/threshold_rmsprop.csv
output_final/threshold_adam.csv
output_final/feature_importance.png
output_final/feature_importance.csv

The model_comparison.csv file contains final test-set metrics. The run_manifest.json file records the random seed, dataset hash, split sizes, selected optimizer, selected thresholds, confusion matrices, and TensorFlow version.
Final verified run
The corrected run reported the following final test metrics:
Model	Threshold	Accuracy	Precision	Recall	F1-score	AUC-ROC
Baseline (SGD, no tuning)	0.50	0.787793	0.617555	0.526738	0.568543	0.832682
Optimized SGD	0.60	0.765082	0.543434	0.719251	0.619102	0.835369
Optimized RMSprop	0.60	0.773598	0.558887	0.697861	0.620690	0.836635
Optimized Adam	0.60	0.771469	0.553498	0.719251	0.625581	0.838851

RMSprop was selected using validation AUC-ROC in this run. Adam achieved the highest test AUC-ROC and F1-score on the particular locked test split. Test performance is reported descriptively and was not used to select the optimizer.
Reproducibility
The script uses random seed 42 for Python, NumPy, and TensorFlow. Reproducibility can still be affected by hardware, TensorFlow versions, operating-system libraries, and CPU execution order. The generated run manifest should be retained with the results so that each report can be traced to its input data and execution configuration.
For stronger statistical evidence, repeat the experiment across multiple stratified seeds and report the mean, standard deviation, or confidence intervals.
Responsible use
The model is a decision-support tool for prioritizing retention outreach. It should not be used as the sole basis for punitive pricing, denial of service, or other high-impact customer decisions. Before deployment, evaluate subgroup performance, privacy controls, calibration, drift, campaign capacity, and actual retention uplift.
Feature importance indicates model reliance on predictive variables; it does not demonstrate that a feature causes churn. Correlated variables may divide or obscure permutation importance.
“After the final execution, I reconciled the model-comparison CSV, confusion matrices, threshold results, feature-importance output, training plots, and written narrative so that all reported values came from the same documented run. I also verified that threshold selection used validation predictions and that the final test set was reserved for the locked-threshold evaluation.”
