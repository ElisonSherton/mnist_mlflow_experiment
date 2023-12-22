from data import *
from model import *
from utils import *
import time
from mlflow import MlflowClient
import mlflow, random, git

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
MLFLOW_TRACKING_HOST = "http://127.0.0.1"
MLFLOW_TRACKING_PORT = "5000"
ENABLE_SYSTEM_LOGGING = True
EXP_NAME = "mlflow_features_exploration"
EXP_TAGS = {"description": "Explore MLFlow Features",
            "purpose": "To check if MLFlow can be used as the default Experiment tracking platform for Perfios DS Team"}

if ENABLE_SYSTEM_LOGGING:
    mlflow.enable_system_metrics_logging()

class Trainer:
    def __init__(self, exp_name, exp_tags = {}, steps_per_epoch = 100, epochs=10, batch_size=64, num_workers=4, lr = 0.0001, mlflow_args = {}):

        self.model = None
        
        self.global_train_step = 0
        self.global_validate_step = 0
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self._init_mlflow(exp_name, exp_tags)

    def _init_mlflow(self, experiment_name, experiment_tags):
        # Create an mlflow client and set the tracking URI
        self.mlflow_client = MlflowClient(tracking_uri=f"{MLFLOW_TRACKING_HOST}:{MLFLOW_TRACKING_PORT}")
        mlflow.set_tracking_uri(f"{MLFLOW_TRACKING_HOST}:{MLFLOW_TRACKING_PORT}")

        # Get or create an experiment object
        experiment = self.mlflow_client.get_experiment_by_name(experiment_name)
        if not experiment:
            self.mlflow_client.create_experiment(name = experiment_name, tags = experiment_tags)
            experiment = self.mlflow_client.get_experiment_by_name(experiment_name)
        
        # Store this experiment object as a variable in the trainer object
        self.experiment = experiment

    def _log_git_state(self):
        if mlflow.active_run():
            repo = git.Repo(search_parent_directories=True)
            repo_working_directory = repo.working_dir
            commit_hash = repo.head.object.hexsha
            branch = repo.active_branch.name

            # Log Git information to MLflow
            mlflow.log_param("working_directory", repo_working_directory)
            mlflow.log_param("commit_hash", commit_hash)
            mlflow.log_param("branch", branch)
            
    def _train_batch(self):
        loss = (self.epochs * self.steps_per_epoch - self.global_train_step) / 100
        if mlflow.active_run():
            mlflow.log_metric("Train/Loss", loss, self.global_train_step)
        
    def _validate_batch(self):
        loss = (self.epochs * self.steps_per_epoch - self.global_train_step) / self.steps_per_epoch
        loss = (self.epochs - loss) / 10
    
    def _train_epoch(self, epoch):
        for idx in range(1, self.steps_per_epoch + 1):
            self.global_train_step += 1
            self._train_batch()
        
        acc_metric = (epoch / self.epochs * 100 + random.randrange(-10, 10)) * 0.01
        acc_metric = min(acc_metric, 1)
        if mlflow.active_run():
            mlflow.log_metric("Train/Accuracy", acc_metric, epoch)


    def _validate_epoch(self, epoch):
        with torch.no_grad():
            for idx in range(1, self.steps_per_epoch + 1):
                self.global_validate_step += 1
                self._validate_batch()
        
        acc_metric = (epoch / self.epochs * 100 + random.randrange(-20, 10)) * 0.01
        acc_metric = min(acc_metric, 1)
        if mlflow.active_run():
            mlflow.log_metric("Validation/Accuracy", acc_metric, epoch)


    def train(self, run_tags = {}):

        # Create an MLFlow run to track all the important things
        # run = self.mlflow_client.create_run(self.experiment.experiment_id, tags = run_tags,)
        run = mlflow.start_run(experiment_id=self.experiment.experiment_id, tags = run_tags)
        self._log_git_state()
        parent_run = run

        for epoch in range(1, self.epochs + 1):
            
            start = time.time()
            time.sleep(1 + random.random())
            self._train_epoch(epoch)
            epoch_training_time = round(time.time() - start, 2)     
            if mlflow.active_run():
                mlflow.log_metric("Train/Epoch_Time", epoch_training_time, epoch)       

            start = time.time()
            time.sleep(1 + random.random())
            self._validate_epoch(epoch)
            epoch_validation_time = round(time.time() - start, 2)
            if mlflow.active_run():
                mlflow.log_metric("Valid/Epoch_Time", epoch_validation_time, epoch)
            
    def test(self):
        pass

learner = Trainer(EXP_NAME, EXP_TAGS)
learner.train(run_tags = {"intent": "manually log git-repo information"})