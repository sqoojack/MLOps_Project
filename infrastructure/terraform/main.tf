provider "aws" {
  region = "us-east-1"
}

# 新增訓練用的 ECR 倉庫
resource "aws_ecr_repository" "train_repo" {
  name = "recsys-train"
  force_delete = true
}

# 新增 EC2 執行所需的 Role (讓 EC2 能讀寫 S3 與 ECR)
resource "aws_iam_role" "ec2_train_role" {
  name = "ec2_train_role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{ Action = "sts:AssumeRole", Effect = "Allow", Principal = { Service = "ec2.amazonaws.com" } }]
  })
}

resource "aws_iam_role_policy_attachment" "ec2_s3_full" {
  role       = aws_iam_role.ec2_train_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

resource "aws_iam_role_policy_attachment" "ec2_ecr_read" {
  role       = aws_iam_role.ec2_train_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_instance_profile" "ec2_train_profile" {
  name = "ec2_train_profile"
  role = aws_iam_role.ec2_train_role.name
}

# Amazon ECR
resource "aws_ecr_repository" "api_repo" {
  name                 = "recsys-api"
  image_tag_mutability = "MUTABLE"
  force_delete = true
}

resource "aws_ecr_repository" "ui_repo" {
  name                 = "recsys-ui"
  image_tag_mutability = "MUTABLE"
  force_delete = true
}

# 2. 網路與 ALB
data "aws_vpc" "default" { default = true }
data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
  filter {
    name   = "availability-zone"
    values = ["us-east-1d"]
  }
}


resource "aws_security_group" "alb_sg" {
  name        = "recsys-alb-sg"
  vpc_id      = data.aws_vpc.default.id
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_lb" "recsys_alb" {
  name               = "recsys-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_sg.id]
  subnets            = data.aws_subnets.default.ids
}

resource "aws_lb_target_group" "api_tg" {
  name        = "recsys-api-tg"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = data.aws_vpc.default.id
  target_type = "ip"
  health_check { path = "/" }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.recsys_alb.arn
  port              = "80"
  protocol          = "HTTP"
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api_tg.arn
  }
}

# 3. ECS Fargate (API)
resource "aws_ecs_cluster" "recsys_cluster" {
  name = "recsys-cluster"
}

resource "aws_iam_role" "ecs_task_execution_role" {
  name = "ecsTaskExecutionRoleRecSys"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{ Action = "sts:AssumeRole", Effect = "Allow", Principal = { Service = "ecs-tasks.amazonaws.com" } }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution" {
  role       = aws_iam_role.ecs_task_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "ecs_sagemaker_policy" {
  name = "ecs-sagemaker-invoke"
  role = aws_iam_role.ecs_task_execution_role.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{ Action = ["sagemaker:InvokeEndpoint"], Effect = "Allow", Resource = "*" }]
  })
}

resource "aws_ecs_task_definition" "api_task" {
  family                   = "recsys-api-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"
  memory                   = "1024"
  execution_role_arn       = aws_iam_role.ecs_task_execution_role.arn
  task_role_arn            = aws_iam_role.ecs_task_execution_role.arn

  container_definitions = jsonencode([{
    name      = "api"
    image     = "${aws_ecr_repository.api_repo.repository_url}:latest"
    essential = true
    portMappings = [{ containerPort = 8000, hostPort = 8000 }]
    environment = [
      # { name = "SAGEMAKER_ENDPOINT_NAME", value = aws_sagemaker_endpoint.recsys_endpoint.name },
      { name = "AWS_REGION", value = "us-east-1" },
      # Redis 需指向你的 ElastiCache 或外部 Redis 服務
      { name = "REDIS_HOST", value = "YOUR_REDIS_ENDPOINT" } 
    ]
  }])
}

resource "aws_security_group" "ecs_sg" {
  name   = "recsys-ecs-sg"
  vpc_id = data.aws_vpc.default.id
  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb_sg.id]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_ecs_service" "api_service" {
  name            = "recsys-api-service"
  cluster         = aws_ecs_cluster.recsys_cluster.id
  task_definition = aws_ecs_task_definition.api_task.arn
  desired_count   = 2
  launch_type     = "FARGATE"
  network_configuration {
    subnets          = data.aws_subnets.default.ids
    security_groups  = [aws_security_group.ecs_sg.id]
    assign_public_ip = true
  }
  load_balancer {
    target_group_arn = aws_lb_target_group.api_tg.arn
    container_name   = "api"
    container_port   = 8000
  }
}

# 4. Amazon SageMaker Endpoint (PyTorch)
resource "aws_iam_role" "sagemaker_execution_role" {
  name = "sagemaker_execution_role_recsys"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{ Action = "sts:AssumeRole", Effect = "Allow", Principal = { Service = "sagemaker.amazonaws.com" } }]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_iam_role_policy_attachment" "ec2_cloudwatch_logs" {
  role       = aws_iam_role.ec2_train_role.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess"
}

# resource "aws_sagemaker_model" "recsys_model" {
#   name               = "recsys-transformer-model"
#   execution_role_arn = aws_iam_role.sagemaker_execution_role.arn

#   primary_container {
#     # 使用 AWS 官方的 PyTorch 映像檔 (對應版本需與你訓練環境相符)
#     image          = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.0.0-cpu-py310-ubuntu20.04-ec2"
#     model_data_url = "s3://YOUR-S3-BUCKET/model.tar.gz"
#   }
# }

# resource "aws_sagemaker_endpoint_configuration" "recsys_endpoint_config" {
#   name = "recsys-endpoint-config"

#   production_variants {
#     variant_name           = "AllTraffic"
#     model_name             = aws_sagemaker_model.recsys_model.name
#     initial_instance_count = 1
#     instance_type          = "ml.m5.large" # 由於是 Transformer，選擇具備一定運算能力的實例
#   }
# }

# resource "aws_sagemaker_endpoint" "recsys_endpoint" {
#   name                 = "recsys-endpoint"
#   endpoint_config_name = aws_sagemaker_endpoint_configuration.recsys_endpoint_config.name
# }



output "current_subnet_id" {
  value = data.aws_subnets.default.ids[0]
}

output "current_sg_id" {
  value = aws_security_group.ecs_sg.id
}