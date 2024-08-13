from source.enums import StockExchange, Target, TMode
from source.packages import install_packages


# Configuration variables for the data to use
ETFS            = ["XLP.US", "XLK.US"]      # List of ETFs to be used in the analysis
START_DATE      = "2021-01-01"              # Start date of the interval
END_DATE        = "2022-12-31"              # End date of the interval
CALENDAR        = StockExchange.NYSE        # Stock exchange calendar to be used                # OPTIONS: NYSE, LSE, EUREX
TARGET          = Target.D3                 # Target to predict on                              # OPTIONS: D3, W1, W2, M1, M3


# Configuration variables for the BERT model, IMPORTANT: The variables will only be used if the model is not already trained 
LEARNING_RATE   = 1e-5                      # Learning rate for the model
EPOCHS          = 3                         # Number of epochs to train the model
BATCH_SIZE      = 16                        # Batch size for the model
WEIGHT_DECAY    = 0.015                     # Weight decay for the model        
MODEL_TYPE      = "distilbert-base-uncased" # Type of the model to be used                      # OPTIONS: "distilbert-base-uncased", "bert-base-uncased", "ProsusAI/finbert"
MODEL_NAME      = f"DBERT_{TARGET}_V1"      # Name of the model to be trained / loaded          # Naming convention: type_target_version


# Configuration variables for the simulation / evaluation
SAMPLE_SIZE     = 1000                      # Number of samples to be drawn from the dataset    # Has to be >= 100, None equals the whole dataset
RANDOM_SEED     = 43                        # Random seed for sample size and reproducibility
THRESH_MODE     = TMode.PERCENTAGE          # Threshold calculation mode to be used             # OPTIONS: Mode.PERCENTAGE, Mode.STATIC, MODE.NORMAL_DISTRIBUTION
THRESHOLD       = 10                        # Threshold value to be used                        # For P: 10 = Top 10% of predictions and Bottom 10% of predictions
NUM_PREDICTIONS = 100                       # Amount of best/worst predictions to be displayed


# Configuration variables for the execution
NOTEBOOK_FILE   = "nb_practicalproject.ipynb"                                                      # Path to the Jupyter notebook file
SAVE_NAME       = f"{TARGET}_{SAMPLE_SIZE}_{THRESH_MODE}_{THRESHOLD}.html"                      # Naming convention: label_size_mode_threshold_html


# Execution of the notebook
if __name__ == "__main__":
    install_packages("requirements.txt", True)

    # delayed import after packages have been installed
    from source.execution import execute_notebook
    execute_notebook(NOTEBOOK_FILE, SAVE_NAME)