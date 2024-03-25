# Waymo Dataset Preparation

Download and preprocess the [Waymo Open Motion Dataset](https://waymo.com/open/) according to the [instructions in the MTR repository](https://github.com/sshaoshuai/MTR/blob/master/docs/DATASET_PREPARATION.md), and organize the data as follows: 
```
EDA
├── MTR
│   ├── data
|   │   ├── waymo
|   │   │   ├── processed_scenarios_training
|   │   │   ├── processed_scenarios_validation
|   │   │   ├── processed_scenarios_training_infos.pkl
|   │   │   ├── processed_scenarios_val_infos.pkl
|   │   │   ├── ...
│   ├── ...
├── ...

```
