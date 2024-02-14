from data import *
from model import *
from utils import *
import time, os
from mlflow import MlflowClient
import mlflow, random, git


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
MLFLOW_TRACKING_PORT = ""
ENABLE_SYSTEM_LOGGING = True
EXP_NAME = "temporary"
EXP_TAGS = {"description": "Explore MLFlow Features",
            "purpose": "To check if MLFlow can be used as the default Experiment tracking platform for Perfios DS Team"}
PARENT_RUN = "vinayak_14_feb_21_13_parent_run"


if ENABLE_SYSTEM_LOGGING:
    mlflow.enable_system_metrics_logging()

class Trainer:
    def __init__(self, exp_name, exp_tags = {}, steps_per_epoch = 10, epochs=3, batch_size=64, num_workers=4, lr = 0.0001, mlflow_args = {}):

        self.model = None
        
        self.global_train_step = 0
        self.global_validate_step = 0
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self._init_mlflow(exp_name, exp_tags)

    def _init_mlflow(self, experiment_name, experiment_tags):
        # Create an mlflow client and set the tracking URI
        self.mlflow_client = MlflowClient(tracking_uri=f"{MLFLOW_TRACKING_HOST}")#:{MLFLOW_TRACKING_PORT}")
        mlflow.set_tracking_uri(f"{MLFLOW_TRACKING_HOST}")#:{MLFLOW_TRACKING_PORT}")

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

    def hyperparam_tune(self, run_tags = {}, nested = False):
        # Check with different epochs configuration i.e. 2, 5, 10 epochs respectively
        run = mlflow.start_run(experiment_id = self.experiment.experiment_id, tags = run_tags, nested = nested)
        for idx, epochs in enumerate([5, 8, 10], start = 1):
            self.epochs = epochs
            self.train(run_tags = {"stage": f"tuning experiment #{idx} with {epochs} epochs", "intent": "Check how many epochs are best for training"}, nested = True)
        mlflow.end_run()

    def train(self, run_tags = {}, nested = False):

        # Create an MLFlow run to track all the important things
        # run = self.mlflow_client.create_run(self.experiment.experiment_id, tags = run_tags,)
        
        run = mlflow.start_run(experiment_id=self.experiment.experiment_id, tags = run_tags, nested = nested)
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
        
        # End the run which was created above
        mlflow.end_run()
            
    def test(self):
        pass

learner = Trainer(EXP_NAME, EXP_TAGS)

# Search all the runs in the given experiment
runs = mlflow.search_runs([learner.experiment.experiment_id])

# Seek if the run exists and get the run ID
existing_run_id = runs[runs["tags.mlflow.runName"] == PARENT_RUN]
rid = ""
if len(existing_run_id) > 0:
    rid = existing_run_id.run_id.iloc[0]

if rid ! = "":
    run = mlflow.start_run(experiment_id = learner.experiment.experiment_id, run_id = rid,
                           tags = {"stage": "main_experiment", "intent": "Train a whole pipeline with multiple steps i.e. hyperparam tuning and then training with best params"}, nested = False)
else:
    run = mlflow.start_run(experiment_id = learner.experiment.experiment_id, run_id = rid,
                           tags = {"stage": "main_experiment", "intent": "Train a whole pipeline with multiple steps i.e. hyperparam tuning and then training with best params"}, nested = False)
    

learner.hyperparam_tune(run_tags = {"stage": "tuning_parameters", "intent": "check hyperparameter tuning for nesting runs"}, nested = True)

learner.train(run_tags = {"stage": "final_training", "intent": "create final model with best parameters"}, nested = True)

# End the top level run
mlflow.end_run()
