# MLOps 專案

如何使用 Git 與 DVC 來進行資料版本控制與建立機器學習 pipeline。

## 前置任務
- 創conda環境, python=3.12
- ```pip install -r requirements.txt```

## 使用Dataset:
公開Dataset: **RetailRocket Dataset**

來源: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset


## 0. 程式碼結構

```
├─ data/raw/    # 放置原始 RetailRocket CSV
├─ features/    # 中間資料（預處理後）
├─ models/  # 訓練後模型存放處
├─ src/     # 所有 Python 腳本
├─ api/     # FastAPI 服務
├─ airflow/     # DAG 放這裡
├─ dvc.yaml     # DVC pipeline 定義（自動建立）
└─ requirements.txt     # 套件依賴
```

## 1. 初始化 Git 與 DVC

```
git init    # 初始化 Git 版本控制（只要做一次）

dvc init    # 初始化 DVC 專案（只要做一次）
```

## 2. 將原始資料events 加入 DVC 管理
```
# 將原始資料 events.csv 加入 DVC 追蹤
dvc add data/raw/events.csv

# 將 DVC metadata檔案加入 Git 版本控制
git add data/raw/.gitignore data/raw/events.csv.dvc
git commit -m "Add raw events.csv with DVC tracking"
```

## 3. 建立資料前處理 pipeline 階段
```
# 新增 pipeline 階段
# -n:  命名為 preprocess
# -d:  依賴檔案
# -o:  輸出檔案
dvc stage add -n preprocess -d src/features.py -d data/raw/events.csv -o features/events_processed.csv python src/features.py


# 執行 pipeline 階段，產生輸出檔案
dvc repro


# 將 pipeline 定義檔加入 Git
git add dvc.yaml dvc.lock
git commit -m "Add preprocess stage to DVC pipeline"
```

## 4. 建立訓練模型pipeline階段
```
# 新增訓練階段，並命名為 train
# 依賴前處理後資料 及 訓練程式碼，輸出模型檔案
dvc stage add -n train -d features/events_processed.csv -d src/train.py -o models/model.pkl python src/train.py

# 執行全部 pipeline（preprocess + train）
dvc repro

# 將 pipeline 定義與鎖定檔加入 Git
git add dvc.yaml dvc.lock
git commit -m "Add train stage to DVC pipeline"
```

# 查看模型
因為已經有MLflow來記錄模型, 所以可以用MLflow UI來檢視訓練紀錄和model資訊
```
mlflow ui
```

## 5. 評估模型 (建立evaluate階段)
```
dvc stage add -n evaluate -d models/popular_items.pkl -d data/raw/events.csv -d src/evaluate.py -o metrics/metrics.json python src/evaluate.py
dvc repro   # 執行整個流程 (包含前面的預訓練和train階段)
```



# Remarks:
- 關於還沒用到的 CSV (ex: category_tree.csv) 
如果目前 pipeline 還沒有用到，暫時不必用 dvc add 加入 DVC 管理。
DVC 最重要是管理「大型且需要版本追蹤的資料」，如果你未來要用，再用 dvc add 加入即可

- 如何用 DVC 管理還沒用的資料?
```
dvc add data/raw/category_tree.csv
git add data/raw/category_tree.csv.dvc   # 用dvc檔去追蹤data
git commit -m "Add category_tree.csv to DVC"
```
- **dvc.lock:**
是 DVC 用來「鎖定」和「記錄」當前 pipeline 各階段輸入、輸出和指令的狀態檔。
