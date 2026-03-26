# src/train_by_AWS.py
import os
import yaml
import sagemaker
import tarfile
import shutil
import hashlib
import boto3
import json
import time
from botocore.exceptions import ClientError
from sagemaker.pytorch import PyTorch
from sagemaker.s3 import S3Downloader
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

def get_local_file_hash(file_path):
    """計算本地檔案的 MD5 Hash"""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def upload_to_s3(local_path, s3_bucket, s3_prefix, sagemaker_session):
    """智慧上傳：比對 Hash，若檔案一致則跳過上傳"""
    s3_client = boto3.client('s3')
    filename = local_path.split('/')[-1]
    s3_key = f"{s3_prefix}/{filename}"
    local_hash = get_local_file_hash(local_path)
    
    try:
        response = s3_client.head_object(Bucket=s3_bucket, Key=s3_key)
        s3_hash = response['ETag'].strip('"')
        if local_hash == s3_hash:
            print(f"檔案已存在且一致，跳過上傳: {s3_key}")
            return f"s3://{s3_bucket}/{s3_key}"
    except ClientError:
        pass

    print(f"正在上傳: {filename}...")
    return sagemaker_session.upload_data(path=local_path, bucket=s3_bucket, key_prefix=s3_prefix)

def wait_for_training_start(log_group, log_stream, region):
    logs = boto3.client('logs', region_name=region)
    pattern = "Train start" # 改成你自訂的字串
    start_time = time.time()
    
    while True:
        try:
            response = logs.get_log_events(
                logGroupName=log_group,
                logStreamName=log_stream,
                startFromHead=True
            )
            events = response.get('events', [])
            for event in events:
                # 建議用 in 來判斷，避免 tqdm 控制字元的干擾
                if pattern in event['message']:
                    print(f"\n 偵測到訓練log: 訓練已正式開始！")
                    return True
            
            elapsed = int(time.time() - start_time)
            print(f"\r⏳ 已等待 {elapsed} 秒... 環境初始化中", end="")
            
        except ClientError:
            pass
        
        # 縮短檢查間隔到 5 秒
        time.sleep(5) 
        
        if time.time() - start_time > 600: 
            return False

def run_ec2_mode(params, bucket, train_s3, test_s3, item_map_s3, params_s3):
    """
    同步執行的 EC2 訓練模式：下載資料 -> 訓練 -> 上傳模型 -> 關機
    """
    ec2 = boto3.client('ec2')
    
    # 從 params 與 env 讀取設定
    image_id = params['train'].get('ec2_ami_id')
    instance_type = params['train'].get('ec2_instance_type', 't3.medium')
    subnet_id = os.getenv('TRAIN_SUBNET_ID') 
    security_group_id = os.getenv('TRAIN_SG_ID')
    key_name = os.getenv('AWS_KEY_NAME')
    ecr_image_uri = os.getenv('ECR_TRAIN_IMAGE_URI')
    region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
    
    # 新增：讀取 MLflow 與 S3 的相關環境變數，準備傳入容器
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', '')
    mlflow_user = os.getenv('MLFLOW_TRACKING_USERNAME', '')
    mlflow_pass = os.getenv('MLFLOW_TRACKING_PASSWORD', '')
    s3_model_path = os.getenv('S3_MODEL_PATH', 'recsys/model_output')

    script_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'ec2_user_data.sh')
    with open(script_path, 'r') as f:
        user_data_template = f.read()
        
    user_data_script = user_data_template.format(
        train_s3=train_s3,
        test_s3=test_s3,
        item_map_s3=item_map_s3,
        params_s3=params_s3,
        region=region,
        ecr_registry=ecr_image_uri.split('/')[0],
        ecr_image_uri=ecr_image_uri,
        bucket=bucket,
        s3_model_path=s3_model_path,
        mlflow_uri=mlflow_uri,
        mlflow_user=mlflow_user,
        mlflow_pass=mlflow_pass
    )

    try:
        print(f"[1/3] 正在啟動 EC2 實例 ({instance_type})...")
        response = ec2.run_instances(
            ImageId=image_id,
            InstanceType=instance_type,
            MinCount=1, MaxCount=1,
            KeyName=key_name,
            BlockDeviceMappings=[
                {
                    # 原本的Amazon Linux 的系統碟路徑是 '/dev/xvda'換成Ubuntu ->要改成 /dev/sda1
                    'DeviceName': '/dev/sda1',
                    'Ebs': {
                        'VolumeSize': 100,      # 給Docker那些的包裝加大到 30GB
                        'VolumeType': 'gp3',
                        'DeleteOnTermination': True 
                    }
                }
            ],
            
            NetworkInterfaces=[{
                'DeviceIndex': 0,
                'SubnetId': subnet_id,
                'Groups': [security_group_id],
                'AssociatePublicIpAddress': True
            }],
            
            InstanceMarketOptions={
                'MarketType': 'spot',   # 減少成本
                'SpotOptions': {
                    'MaxPrice': '0.4', # 你願意支付的最高時薪，不填則預設為 On-Demand 價格
                    'SpotInstanceType': 'one-time' # 執行一次，被收回後不會自動重啟
                }
            },
            
            UserData=user_data_script,
            InstanceInitiatedShutdownBehavior='terminate',
            IamInstanceProfile={'Name': 'ec2_train_profile'},
            TagSpecifications=[{'ResourceType': 'instance', 'Tags': [{'Key': 'Name', 'Value': 'MLOps-Training-Job'}]}]
        )
        
        instance_id = response['Instances'][0]['InstanceId']
        
        # 1. 等待開機完成
        print(f"[2/3] 等待實例 {instance_id} 進入運行狀態...")
        ec2.get_waiter('instance_running').wait(InstanceIds=[instance_id])
        log_group = "/aws/ec2/MLOps-TrainingJobs"
        wait_for_training_start(log_group, instance_id, region)

        print(f"⏳ 正在監控 S3 模型檔案是否產生...")
        s3 = boto3.client('s3', region_name=region)
        s3_waiter = s3.get_waiter('object_exists')
        
        model_s3_key = "recsys/model_output/model.tar.gz"
        
        try:
            # 每 20 秒檢查一次 S3，最多等待 45 分鐘 (135 * 20s)
            s3_waiter.wait(
                Bucket=bucket,
                Key=model_s3_key,
                WaiterConfig={'Delay': 20, 'MaxAttempts': 135}
            )
            print(f"🚀 偵測到模型檔案已上傳至 S3，訓練任務順利結束")
        except Exception as e:
            print(f"❌ 等待 S3 模型檔案超時或失敗: {e}")
            raise e

        # 3. 下載模型成果
        print(f"[3/3] 正在從 S3 下載訓練成果...")
        model_s3_uri = f"s3://{bucket}/recsys/model_output/model.tar.gz"
        download_and_extract_model(model_s3_uri, params)
        
        return instance_id

    except ClientError as e:
        print(f"❌ EC2 流程失敗: {e}")
        raise e

def run_sagemaker_mode(params, bucket, role, sagemaker_session, train_s3, test_s3, item_map_s3):
    """原本的 SageMaker 訓練模式邏輯"""
    print("🛠️ [SageMaker Mode] 提交訓練任務...")
    estimator = PyTorch(
        entry_point='src/train.py',
        source_dir='.',
        role=role,
        framework_version='2.1.0',
        py_version='py310',
        instance_count=1,
        instance_type=params['train'].get('instance_type', 'ml.m5.large'),
        sagemaker_session=sagemaker_session,
        output_path=f"s3://{bucket}/recsys/model_output",
        environment={
            'MLFLOW_TRACKING_URI': os.getenv('MLFLOW_TRACKING_URI'),
            'MLFLOW_TRACKING_USERNAME': os.getenv('MLFLOW_TRACKING_USERNAME'),
            'MLFLOW_TRACKING_PASSWORD': os.getenv('MLFLOW_TRACKING_PASSWORD'),
        }
    )

    estimator.fit({                                 # 只有 1 層：字典 {}
    'train': train_s3, 
    'test': test_s3, 
    'item_map': item_map_s3
    })
    return estimator.model_data

def download_and_extract_model(model_s3_uri, params):
    s3 = boto3.client('s3')
    tmp_dir = "tmp_model"
    os.makedirs(tmp_dir, exist_ok=True)
    
    # 解析 S3 URI (例如 s3://bucket/path/to/file)
    bucket_name = model_s3_uri.split('/')[2]
    key = '/'.join(model_s3_uri.split('/')[3:])
    tar_path = os.path.join(tmp_dir, "model.tar.gz")

    print(f"正在從 S3 下載模型檔案: {model_s3_uri}...")
    s3.download_file(bucket_name, key, tar_path) # 使用純 boto3 下載
    
    if os.path.exists(tar_path):
        with tarfile.open(tar_path, "r:gz") as tar:
            target_output_path = params['data']['model_path']
            os.makedirs(os.path.dirname(target_output_path), exist_ok=True)
            tar.extract("model.pth", path=tmp_dir)
            shutil.move(os.path.join(tmp_dir, "model.pth"), target_output_path)
        print(f"✅ 模型已成功同步至本地路徑: {target_output_path}")
    shutil.rmtree(tmp_dir)

def main():
    # 1. 取得與驗證環境變數
    role = os.getenv('SAGEMAKER_ROLE_ARN')
    bucket = os.getenv('S3_BUCKET_NAME')
    
    if not all([role, bucket]):
        raise ValueError("缺少必要環境變數: SAGEMAKER_ROLE_ARN 或 S3_BUCKET_NAME")

    sagemaker_session = sagemaker.Session(default_bucket=bucket)

    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    # 2. 同步資料至 S3 (兩種模式都需要資料在雲端)
    print("📦 開始同步設定檔及訓練資料至 S3...")
    train_s3 = upload_to_s3(params['data']['processed_train_path'], bucket, "recsys/data/train", sagemaker_session)
    test_s3 = upload_to_s3(params['data']['processed_test_path'], bucket, "recsys/data/test", sagemaker_session)
    item_map_s3 = upload_to_s3(params['data']['item_map_path'], bucket, "recsys/data/item_map", sagemaker_session)
    params_s3 = upload_to_s3("params.yaml", bucket, "recsys/config", sagemaker_session)

    # 3. 根據 mode 執行對應流程
    mode = params['train'].get('mode', 'sagemaker')
    
    if mode == "sagemaker":
        model_data_uri = run_sagemaker_mode(params, bucket, role, sagemaker_session, train_s3, test_s3, item_map_s3)
        # 4. 下載模型成果
        download_and_extract_model(model_data_uri, params)
        
    elif mode == "ec2":
        run_ec2_mode(params, bucket, train_s3, test_s3, item_map_s3, params_s3)
        print("EC2 同步訓練與模型下載完成。")
        
    else:
        raise ValueError(f"不支援的模式: {mode}")

if __name__ == "__main__":
    main()