#!/bin/bash

#SBATCH -N 1               # Number of nodes
#SBATCH -c 4               # Number of cores
#SBATCH -t 0-06:00:00      # Time in d-hh:mm:ss
#SBATCH -p general         # Partition
#SBATCH -q public          # QOS
#SBATCH --gpus=1      # GPU specification
#SBATCH --mem=256GB         # Memory allocation
#SBATCH -o slurm.%j.out    # File to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err    # File to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL    # Send an email when the job starts, stops, or fails
#SBATCH --export=NONE      # Purge the job-submitting shell environment

# Load necessary modules
module load mamba/latest

# Check if the Conda environment exists; create it if missing
ENV_NAME="tcrgpepi_1000_1"
if ! conda info --envs | grep -q "$ENV_NAME"; then
    echo "Creating Conda environment '$ENV_NAME'..."
    mamba create --name $ENV_NAME python=3.8 -y
fi

# Activate the Conda environment
echo "Activating Conda environment '$ENV_NAME'..."
source activate $ENV_NAME


echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install required Python libraries
echo "Installing required Python libraries..."
pip install pandas==1.4.3 numpy==1.21.6 scikit-learn==1.1.2 imbalanced-learn==0.9.1 psutil --quiet

# Install PyTorch (ensure compatibility with GPU if available)
echo "Installing PyTorch with GPU support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet


# Verify if required files exist
TRAIN_FILE="train.csv"
TEST_FILE="test.csv"

if [[ ! -f "$TRAIN_FILE" || ! -f "$TEST_FILE" ]]; then
    echo "Error: '$TRAIN_FILE' or '$TEST_FILE' not found. Please make sure both files are in the current directory."
    exit 1
fi


echo "Script execution complete!"
# Run your Python script, pointing to the dataset file on the cluster
python tcrgp.py
