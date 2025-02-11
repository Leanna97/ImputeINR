# ImputeINR: Time Series Imputation via Implicit Neural Representations for Disease Diagnosis with Missing Data

<p align="center">
<img src=plot/INR_for_TS.png width="500" height="300"/>
</p>

## Abstract
Healthcare data frequently contain a substantial proportion of missing values, necessitating effective time series imputation to support downstream disease diagnosis tasks. However, existing imputation methods focus on discrete data points and are unable to effectively model sparse data, resulting in particularly poor performance for imputing substantial missing values. In this paper, we propose a novel approach, ImputeINR, for time series imputation by employing implicit neural representations (INR) to learn continuous functions for time series. ImputeINR leverages the merits of INR in that the continuous functions are not coupled to sampling frequency and have infinite sampling frequency, allowing ImputeINR to generate fine-grained imputations even on extremely sparse observed values. Extensive experiments conducted on eight datasets with five ratios of masked values show the superior imputation performance of ImputeINR, especially for high missing ratios in time series data. Furthermore, we validate that applying ImputeINR to impute missing values in healthcare data enhances the performance of downstream disease diagnosis tasks.

## Overall Architecture
<p align="center">
<img src=plot/architecture.png width="1100" height="500"/>
</p>


## Get Started
Please install Python>=3.8 and install the requirements via:
```
pip install -r requirements.txt
```

Please download the ETT and Weather datasets from [TimesNet](https://github.com/thuml/Time-Series-Library) and store the data in `./all_datasets`.

Please download the Phy2012, Phy2019, BAQ, IAQ, and Solar datasets from [TSDB/PyPOTS](https://github.com/WenjieDu/TSDB) and store the data in `./datasets`.

Please download the MIMIC3 dataset from [MIMIC3 Benchmark](https://github.com/YerevaNN/mimic3-benchmarks/tree/master) and store the data in `./datasets`.

Then please run the ImputeINR method with following command by choosing the configuration from `./cfgs`:
```
CUDA_VISIBLE_DEVICES=0 python run_trainer.py --cfg <path to cfg>
```

For example:
```
CUDA_VISIBLE_DEVICES=0 python run_trainer.py --cfg ./cfgs/Weather.yaml
```


## Imputation Results
<p align="center">
<img src=plot/main_results.png width="1100" height="800"/>
</p>

## Disease Diagnosis Results
<p align="center">
<img src=plot/disease_diagnosis.png width="1000" height="150"/>
</p>

## Efficiency Analysis
<p align="center">
<img src=plot/efficiency_analysis.png width="500" height="350"/>
</p>

