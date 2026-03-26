#!/bin/bash
systemctl start docker

mkdir -p /tmp/input/train /tmp/input/test /tmp/input/item_map /tmp/model
chmod -R 777 /tmp/model

aws s3 cp {train_s3} /tmp/input/train/
aws s3 cp {test_s3} /tmp/input/test/
aws s3 cp {item_map_s3} /tmp/input/item_map/
aws s3 cp {params_s3} /tmp/params.yaml

INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)

aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {ecr_registry}

docker run --gpus all --name training_container \
    --log-driver=awslogs \
    --log-opt awslogs-group=/aws/ec2/MLOps-TrainingJobs \
    --log-opt awslogs-create-group=true \
    --log-opt awslogs-region={region} \
    --log-opt awslogs-stream=$INSTANCE_ID \
    -v /tmp/input/train:/opt/ml/input/data/train \
    -v /tmp/input/test:/opt/ml/input/data/test \
    -v /tmp/input/item_map:/opt/ml/input/data/item_map \
    -v /tmp/params.yaml:/app/params.yaml \
    -v /tmp/model:/opt/ml/model \
    -e SM_CHANNEL_TRAIN=/opt/ml/input/data/train \
    -e SM_CHANNEL_TEST=/opt/ml/input/data/test \
    -e SM_CHANNEL_ITEM_MAP=/opt/ml/input/data/item_map \
    -e SM_MODEL_DIR=/opt/ml/model \
    -e AWS_DEFAULT_REGION={region} \
    -e S3_BUCKET_NAME={bucket} \
    -e S3_MODEL_PATH={s3_model_path} \
    -e MLFLOW_TRACKING_URI={mlflow_uri} \
    -e MLFLOW_TRACKING_USERNAME={mlflow_user} \
    -e MLFLOW_TRACKING_PASSWORD={mlflow_pass} \
    {ecr_image_uri} python src/train.py

cd /tmp/model
if [ -f "model.pth" ]; then
    tar -czf /tmp/model.tar.gz .
    aws s3 cp /tmp/model.tar.gz s3://{bucket}/recsys/model_output/model.tar.gz
else
    echo "Error: model.pth not found in /tmp/model. Training container failed to produce output."
fi
shutdown -h now