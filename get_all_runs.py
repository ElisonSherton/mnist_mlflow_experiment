from data import *
from model import *
from utils import *
import time, os
from mlflow import MlflowClient
import mlflow, random, git

os.environ["MLFLOW_TRACKING_USERNAME"] = "vinayak.n@perfios.com"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "an8H;7K!.p-&@Wd3"
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
MLFLOW_TRACKING_HOST = "https://mlflow.theeigens.com"
MLFLOW_TRACKING_PORT = ""
ENABLE_SYSTEM_LOGGING = True

mlflow.set_tracking_uri(f"{MLFLOW_TRACKING_HOST}")
runs = mlflow.search_runs(["1"])

import pdb; pdb.set_trace();