#!/bin/tcsh

#SBATCH --job-name="test_architecture"
#SBATCH --partition="fge"
#SBATCH --account="rda-ghpcs"
#SBATCH --qos="batch"
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
##SBATCH --nodes=1
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=1
##SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --exclude=h28n12
##SBATCH --exclude=h31n03,h28n12
##SBATCH --exclude=h29n01,h26n05,h28n06,h29n07,h30n03,h30n15
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ryan.lagerquist@noaa.gov
#SBATCH --output=test_architecture_%A.out

module load cuda/10.1
source /scratch2/BMC/gsd-hpcs/Jebb.Q.Stewart/conda3.7/etc/profile.d/conda.csh
conda activate base

set CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_wildfire_wpo_standalone/ml_for_wildfire_wpo"

python3 -u "${CODE_DIR_NAME}/test_architecture.py"