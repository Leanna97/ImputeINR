# ImputeINR: Adaptive Group-based Implicit Neural Representations for Time Series Imputation

<p align="center">
<img src=plot/INR_for_TS.png width="500" height="300"/>
</p>

<p align="center">
<img src=plot/motivation.png width="1100" height="300"/>
</p>

## Abstract
Time series data frequently exhibit the presence of missing values, rendering imputation a crucial process for downstream time series tasks and applications. However, existing imputation methods focus on discrete data points and are unable to effectively model sparse data, resulting in particularly poor performance for imputing substantial missing values. In this paper, we propose a novel approach, ImputeINR, for time series imputation by employing implicit neural representations (INR) to learn continuous functions for time series. ImputeINR leverages the merits of INR in that the continuous functions are not coupled to sampling frequency and have infinite sampling frequency, allowing ImputeINR to generate fine-grained imputations even on extremely sparse observed values. In addition, we introduce a multi-scale feature extraction module in ImputeINR architecture to capture patterns from different time scales, thereby effectively enhancing the fine-grained and global consistency of the imputation. To address the unique challenges of complex temporal patterns and multiple variables in time series, we design a specific form of INR continuous function that contains %three additional components to learn trend, seasonal, and residual information separately. Furthermore, we innovatively propose an adaptive group-based framework to model complex information, where variables with similar distributions are modeled by the same group of multilayer perception layers to extract necessary correlation features. Since the number of groups and their output variables are determined by variable clustering, ImputeINR has the capacity to adapt to diverse datasets. Extensive experiments conducted on seven datasets with five ratios of masked values demonstrate the superior performance of ImputeINR, especially for high missing ratios in time series.


## Reconstruction from INR
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

Then please run the ImputeINR method with following command by choosing the configuration from `./cfgs`:
```
CUDA_VISIBLE_DEVICES=0 python run_trainer.py --cfg <path to cfg>
```

For example:
```
CUDA_VISIBLE_DEVICES=0 python run_trainer.py --cfg ./cfgs/Weather.yaml
```


## Main Results
<p align="center">
<img src=plot/main_results.png width="1100" height="600"/>
</p>

## Robustness Analysis
<p align="center">
<img src=plot/robustness_analysis.png width="1000" height="350"/>
</p>

## Efficiency Analysis
<p align="center">
<img src=plot/efficiency_analysis.png width="500" height="350"/>
</p>

