# Installation

**Step 1:** Clone the codebase.
```shell
git clone https://github.com/Longzhong-Lin/EDA.git
cd EDA
```

**Step 2:** Create the python environment.
```shell
conda create --name eda python=3.8 -y
conda activate eda
pip install -r requirements.txt
```

**Step 3:** [Install MTR](https://github.com/sshaoshuai/MTR/blob/master/docs/INSTALL.md) under the EDA directory as follows.
```
EDA
├── MTR
│   ├── data
│   ├── mtr
│   ├── tools
│   ├── ...
├── eda
├── tools
├── ...
```
