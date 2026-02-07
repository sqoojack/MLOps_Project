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

    # 這裡的指令會在 airflow-scheduler 容器內執行
    # 因為我們掛載了本機目錄到 /opt/airflow/project
    # 所以容器內產生的檔案變更，會直接反映到你的實體硬碟上
    run_pipeline = BashOperator(
        task_id='run_dvc_pipeline',
        bash_command=f'cd {PROJECT_PATH} && dvc repro',
    )

    # 步驟 3: (可選) 推送新模型到 Registry 或 DVC Remote
    push_results = BashOperator(
        task_id='push_model_and_metrics',
        bash_command=f'cd {PROJECT_PATH} && dvc push',
    )

    # 設定依賴關係
    pull_data >> run_pipeline >> push_results