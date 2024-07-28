#!/bin/bash
#SBATCH --job-name=isp_dask_job        # Job name
#SBATCH --output=isp_dask_output.log   # Output file
#SBATCH --error=isp_dask_error.log     # Error file
#SBATCH --ntasks=1                     # Run on a single CPU
#SBATCH --time=02:00:00                # Time limit hrs:min:sec
#SBATCH --mem=4G                       # Memory limit

# Load the necessary modules
module load python/3.11.5

# Activate virtual environment (if any)
source ~/myenv/bin/activate

# Navigate to the directory containing script
cd /path/to/your/script

# Run the Python script
python ISP_Dask.py