# MLOps 推薦系統專案

本專案實作了一個完整的 MLOps 機器學習pipeline，用於構建與部署電子商務推薦系統。專案整合了資料版本控制 (DVC)、排程管理 (Airflow)、模型追蹤 (MLflow)、CI/CD (GitHub Actions)，並提供基於 FastAPI 的後端服務與 Streamlit 前端介面。

## 系統架構與核心技術

* **資料與模型版本控制**: DVC 與 Git
* **任務排程**: Apache Airflow
* **模型追蹤與註冊**: MLflow
* **後端 API**: FastAPI
* **前端 UI**: Streamlit
* **快取與狀態管理**: Redis
* **雲端推論整合**: 支援 AWS SageMaker Runtime
* **持續整合/持續部署 (CI/CD)**: GitHub Actions

## 1. DVC 資料流與 Pipeline 階段

資料集使用 Amazon Beauty Metadata (`data/raw/meta_Beauty.json.gz`) 作為原始輸入資料。
Pipeline 定義於 `dvc.yaml` 中，包含以下三個主要階段：

1.  **preprocess**:
    * 執行指令: `python src/features.py`
    * 輸出: `train.csv`, `test.csv`, `item_map.json`, `items_metadata.json`
2.  **train**:
    * 執行指令: `python src/train.py`
    * 輸出: 訓練完成的模型權重檔 `model.pth`
3.  **evaluate**:
    * 執行指令: `python src/evaluate.py`
    * 輸出指標: `metrics.json`

執行完整 pipeline：
```bash
dvc repro
```

## 2. 服務部署 (Docker Compose)

本專案使用 Docker Compose 管理微服務，定義於 `docker-compose.yaml`。

### 啟動服務
```bash
docker-compose up -d
```

### 服務與通訊埠配置
* **Airflow Webserver**: `http://localhost:8080` (管理 DAGs 與排程)
* **FastAPI API**: `http://localhost:8000` (模型推論與使用者行為紀錄)
* **Streamlit UI**: `http://localhost:8501` (模擬電商互動介面)
* **Redis**: `localhost:6379` (儲存使用者的瀏覽與互動歷史 `user:{user_id}`)

## 3. API 端點說明

FastAPI 提供以下 RESTful 端點：
* `GET /browse`: 隨機回傳商品供使用者瀏覽。
* `POST /interact`: 記錄使用者感興趣的商品 (寫入 Redis)。
* `POST /recommend`: 根據 Redis 中紀錄的使用者互動歷史，回傳個人化推薦結果。若設定了 AWS 相關環境變數，將轉發推論請求至 SageMaker Endpoint。
* `DELETE /history`: 清空特定使用者的互動歷史。

## 4. CI/CD 流程 (GitHub Actions)

專案包含自動化 CI/CD 流程，定義於 `.github/workflows/mlops.yaml`：
1.  **單元測試**: 執行 `pytest tests/`。
2.  **DVC Pipeline**: 當推送到 `main` 分支時，自動執行 `dvc repro` 重新訓練與評估模型。
3.  **指標檢視**: 輸出 `metrics.json` 供驗證。
4.  **模型註冊與晉升**: 透過腳本呼叫 MLflow Client，尋找 NDCG 指標最佳的 Run，建立模型版本並自動晉升為 `Production` 階段。
