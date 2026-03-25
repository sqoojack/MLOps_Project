#!/bin/bash
# push_train_image.sh

# 請替換為你的 AWS 帳號 ID
# ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ACCOUNT_ID=647223940998
REGION="us-east-1"
REPO_NAME="recsys-train"
IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:latest"

echo "🔐 正在登入 ECR..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

echo "📦 正在建置訓練鏡像..."
docker build -t ${REPO_NAME} -f infrastructure/docker/Dockerfile.train .

echo "🏷️ 正在標記鏡像..."
docker tag ${REPO_NAME}:latest ${IMAGE_URI}

echo "🚀 正在推送到 ECR..."
docker push ${IMAGE_URI}

echo "✅ 完成！請確保 .env 中的 ECR_TRAIN_IMAGE_URI 等於: ${IMAGE_URI}"