from data import *
from model import *
from utils import *
import time
from mlflow import MlflowClient
import mlflow, random

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
MLFLOW_TRACKING_HOST = "http://127.0.0.1"
MLFLOW_TRACKING_PORT = "5000"

class Trainer:
    def __init__(self, model, padding, epochs=50, batch_size=64, num_workers=4,
                  lr = 0.0001,  checkpoint_dir="temp", debug=False, mlflow_args = {}):
        

        self.train_loader = None
        self.val_loader = None

        self.model = None

        self.optimizer = None
        self.criterion = None

        self.global_train_step = 0
        self.global_validate_step = 0
        self.epochs = epochs

    def _init_mlflow(self, experiment_name, experiment_tags):
        # Create an mlflow client and set the tracking URI
        self.mlflow_client = MlflowClient(tracking_uri=f"{MLFLOW_TRACKING_HOST}:{MLFLOW_TRACKING_PORT}")
        mlflow.set_tracking_uri(f"{MLFLOW_TRACKING_HOST}:{MLFLOW_TRACKING_PORT}")

        # Get or create an experiment object
        experiment = self.mlflow_client.get_experiment_by_name(experiment_name)
        if not experiment:
            self.client.create_experiment(name = experiment_name, tags = experiment_tags)
            experiment = self.mlflow_client.get_experiment_by_name(experiment_name)
        
        # Store this experiment object as a variable in the trainer object
        self.experiment = experiment

    def _train_batch(self, inputs, labels):
        # Use GPU or CPU depending on the hardware availability
        inputs = inputs.to(DEVICE); labels = labels.to(DEVICE)

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # Forward Propagation
        outputs = self.model(inputs)

        # Loss Computation
        loss = self.criterion(outputs, labels)
        
        # Gradient Computation
        loss.backward()

        # Parameter updates
        self.optimizer.step()
        
    def _validate_batch(self, inputs, labels):
        with torch.no_grad():
            # Use GPU or CPU depending on the hardware availability
            inputs = inputs.to(DEVICE); labels = labels.to(DEVICE)

            # Forward Propagation
            outputs = self.model(inputs)

            # Loss Computation
            loss = self.criterion(outputs, labels)
            
            # Do all your logging here
    
    def _train_epoch(self, epoch):
        for idx, data in enumerate(self.train_loader, start = 1):
            inputs, labels = data
            self.global_train_step += 1
            self._train_batch(inputs, labels)
    
    def _validate_epoch(self, epoch):
        with torch.no_grad():
            for idx, data in enumerate(self.val_loader, start = 1):
                inputs, labels = data
                self.global_validate_step += 1
                self._validate_batch(inputs, labels)

    def train(self, run_tags = {}):

        # Create an MLFlow run to track all the important things
        run = self.mlflow_client.create_run(self.experiment.experiment_id, tags = run_tags)

        for epoch in range(1, self.max_epochs + 1):
            
            start = time.time()
            self._train_epoch(epoch)
            epoch_training_time = round(time.time() - start, 2)            

            start = time.time()
            self._validate_epoch(epoch)
            epoch_validation_time = round(time.time() - start, 2)

            # Log the times for training and validation respectively 
            mlflow.log_param()
            mlflow.log_metric()

    def test(self):
        pass
