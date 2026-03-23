# src/run_sagemaker_training.py
import os
import yaml
import sagemaker
from sagemaker.pytorch import PyTorch
from dotenv import load_dotenv

# 載入 .env 檔案中的環境變數
load_dotenv()

def main():
    # 1. 取得與驗證環境變數
    role = os.getenv('SAGEMAKER_ROLE_ARN')
    bucket = os.getenv('S3_BUCKET_NAME')
    region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')

    if not all([role, bucket]):
        raise ValueError("缺少必要的環境變數：SAGEMAKER_ROLE_ARN 或 S3_BUCKET_NAME")

    # 2. 初始化 SageMaker Session
    sagemaker_session = sagemaker.Session(default_bucket=bucket)

    # 讀取專案參數
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    # 3. 同步訓練資料至 S3
    print("開始同步資料至 S3...")
    prefix = "recsys/data"
    
    # 取得本地端檔案路徑
    local_train_path = params['data']['processed_train_path']
    local_test_path = params['data']['processed_test_path']
    local_item_map_path = params['data']['item_map_path']

    # 上傳至 S3，sagemaker_session.upload_data 回傳的是 S3 URI
    train_s3 = sagemaker_session.upload_data(path=local_train_path, bucket=bucket, key_prefix=f"{prefix}/train")
    test_s3 = sagemaker_session.upload_data(path=local_test_path, bucket=bucket, key_prefix=f"{prefix}/test")
    item_map_s3 = sagemaker_session.upload_data(path=local_item_map_path, bucket=bucket, key_prefix=f"{prefix}/item_map")

    print(f"資料上傳完成:\nTrain: {train_s3}\nTest: {test_s3}\nItem Map: {item_map_s3}")

    # 4. 定義 SageMaker PyTorch 訓練任務
    estimator = PyTorch(
        entry_point='src/train.py',
        source_dir='.',  # 設定為根目錄，以確保 params.yaml 與 src/ 下的其他模組皆被打包進容器
        role=role,
        framework_version='2.0.0', # 需與 main.tf 中定義的版本對應
        py_version='py310',
        instance_count=1,
        instance_type='ml.g4dn.xlarge', # Transformer 模型建議使用 GPU 實例進行訓練
        sagemaker_session=sagemaker_session,
        output_path=f"s3://{bucket}/recsys/model_output",
        environment={
            'MLFLOW_TRACKING_URI': os.getenv('MLFLOW_TRACKING_URI', '')
        }
    )

    # 5. 提交訓練任務
    # 此處定義的 dict key (train, test, item_map) 會對應為容器內的環境變數：
    # SM_CHANNEL_TRAIN, SM_CHANNEL_TEST, SM_CHANNEL_ITEM_MAP
    print("提交 SageMaker Training Job...")
    estimator.fit({
        'train': train_s3,
        'test': test_s3,
        'item_map': item_map_s3
    })

    print(f"訓練完成，模型輸出存放於: {estimator.model_data}")

if __name__ == "__main__":
    main()