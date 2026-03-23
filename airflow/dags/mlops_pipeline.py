from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

# PROJECT_PATH = "/home/jack/MLOps_Project"   # 因為是自動化, 所以需要專案路徑
PROJECT_PATH = "/opt/airflow/project"


default_args = {
    'owner': 'sqoojack',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

airflow_env = os.environ.copy()

# 定義 DAG
# schedule_interval='@daily' 表示每天執行一次 (Near-line Batch)
with DAG(
    'mlops_nearline_retraining',
    default_args=default_args,
    description='Automated Near-line Retraining Pipeline using DVC',
    schedule_interval='@daily', 
    start_date=days_ago(1),
    catchup=False,
    tags=['mlops', 'recommendation'],
) as dag:

    # 步驟 1: 拉取最新資料 (模擬從 Feature Store 或 DB 同步)
    pull_data = BashOperator(
        task_id='pull_data_and_code',
        bash_command=(
        # f'git config --global --add safe.directory {PROJECT_PATH} && '
        # f'cd {PROJECT_PATH} && git pull && dvc pull'
        f'cd {PROJECT_PATH} &&  dvc pull -f'
        ),
    )
    
    # 2. 從 Redis 撈取最新使用者互動事件，寫入 /feature/events_processed.csv
    extract_events = BashOperator(
        task_id='extract_redis_events',
        bash_command=f'cd {PROJECT_PATH} && python src/extract_latest_events.py',
    )

    # 3. 執行 DVC Pipeline
    run_pipeline = BashOperator(
        task_id='run_dvc_pipeline',
        bash_command=f'cd {PROJECT_PATH} && dvc repro',
        env=airflow_env, 
    )

    # 4: 推送新模型到 Registry 或 DVC Remote
    push_results = BashOperator(
        task_id='push_model_and_metrics',
        bash_command=f'cd {PROJECT_PATH} && dvc push',
        env=airflow_env, 
    )

    # 設定依賴關係
    pull_data >> extract_events >> run_pipeline >> push_results