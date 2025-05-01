import mlflow
import yaml

mlflow.set_tracking_uri("http://mlflow-server:5000")  # <- Tambahkan ini

RUN_ID = None  # Global

def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

def start_mlflow_run(role):
    global RUN_ID
    config = load_config()
    mlflow.set_experiment("FedLDA_Experiment")
    run = mlflow.start_run(run_name=f"{role}_run")
    RUN_ID = run.info.run_id  # Simpan run_id

    for section, params in config.items():
        for key, value in params.items():
            mlflow.log_param(f"{section}_{key}", value)

def log_coherence(score, round_id):
    if RUN_ID:
        with mlflow.start_run(run_id=RUN_ID):
            mlflow.log_metric("coherence", score, step=round_id)

def log_phi_top_words(topics, round_id):
    if RUN_ID:
        for tid, words in topics:
            with mlflow.start_run(run_id=RUN_ID):
                mlflow.log_text(", ".join(words), f"round_{round_id}/topic_{tid}.txt")

def end_run():
    mlflow.end_run()

