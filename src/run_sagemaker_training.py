# src/run_sagemaker_training.py
import os
import yaml
import sagemaker
import tarfile
import shutil
from sagemaker.pytorch import PyTorch
from sagemaker.s3 import S3Downloader
from dotenv import load_dotenv

# 載入 .env 檔案中的環境變數
load_dotenv()

def main():
    # 1. 取得與驗證環境變數
    role = os.getenv('SAGEMAKER_ROLE_ARN')
    bucket = os.getenv('S3_BUCKET_NAME')
    region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')

    if not all([role, bucket]):
        raise ValueError("缺少必要的環境變數: SAGEMAKER_ROLE_ARN 或 S3_BUCKET_NAME")

    # 2. 初始化 SageMaker Session
    sagemaker_session = sagemaker.Session(default_bucket=bucket)

    # 讀取專案參數
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    # 3. 同步訓練資料至 S3
    print("開始同步資料至 S3...")
    prefix = "recsys/data"
    
    local_train_path = params['data']['processed_train_path']
    local_test_path = params['data']['processed_test_path']
    local_item_map_path = params['data']['item_map_path']

    train_s3 = sagemaker_session.upload_data(path=local_train_path, bucket=bucket, key_prefix=f"{prefix}/train")
    test_s3 = sagemaker_session.upload_data(path=local_test_path, bucket=bucket, key_prefix=f"{prefix}/test")
    item_map_s3 = sagemaker_session.upload_data(path=local_item_map_path, bucket=bucket, key_prefix=f"{prefix}/item_map")

    print(f"資料上傳完成:\nTrain: {train_s3}\nTest: {test_s3}\nItem Map: {item_map_s3}")

    # 4. 定義 SageMaker PyTorch 訓練任務
    estimator = PyTorch(
        entry_point='src/train.py',
        source_dir='.',
        role=role,
        framework_version='2.0.0',
        py_version='py310',
        instance_count=1,
        instance_type='ml.g4dn.xlarge',
        sagemaker_session=sagemaker_session,
        output_path=f"s3://{bucket}/recsys/model_output",
        environment={
            'MLFLOW_TRACKING_URI': os.getenv('MLFLOW_TRACKING_URI', '')
        }
    )

    # 5. 提交訓練任務
    print("提交 SageMaker Training Job...")
    estimator.fit({
        'train': train_s3,
        'test': test_s3,
        'item_map': item_map_s3
    })

    # === [核心新增：下載並解壓模型] ===
    model_s3_uri = estimator.model_data
    print(f"訓練完成，模型輸出存放於: {model_s3_uri}")

    tmp_dir = "tmp_model"
    os.makedirs(tmp_dir, exist_ok=True)
    
    print(f"正在從 S3 下載模型檔案...")
    S3Downloader.download(model_s3_uri, tmp_dir, sagemaker_session=sagemaker_session)
    
    tar_path = os.path.join(tmp_dir, "model.tar.gz")
    if os.path.exists(tar_path):
        print(f"正在解壓縮模型檔案...")
        with tarfile.open(tar_path, "r:gz") as tar:
            # 我們只需要 model.pth
            # 根據 params.yaml 的定義：artifacts/model.pth
            target_output_path = params['data']['model_path']
            target_dir = os.path.dirname(target_output_path)
            os.makedirs(target_dir, exist_ok=True)
            
            # 在 tar 檔中尋找 model.pth 並解壓
            tar.extract("model.pth", path=tmp_dir)
            shutil.move(os.path.join(tmp_dir, "model.pth"), target_output_path)
            
        print(f"✅ 模型已成功同步至本地路徑: {target_output_path}")
    
    # 清理臨時資料夾
    shutil.rmtree(tmp_dir)
    # ===============================

if __name__ == "__main__":
    main()