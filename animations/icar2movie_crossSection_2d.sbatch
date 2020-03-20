#! /bin/bash
#SBATCH -J IcarAnim                        # Job name
#SBATCH --account p48500028
#SBATCH -o hist.log                     # Standard output and error log
#SBATCH -e hist.err
#SBATCH -t 24:00:00                       # Time limit hrs:min:sec
#SBATCH -p dav			          # resources group under which you would like to schedule your job
#SBATCH -N 1                              # Number of nodes that you need
#SBATCH --mem 50G
#SBATCH -n 1                              # Run on a single CPU

### Set temp to scratch
export TMPDIR /glade/scratch/currierw/temp
mkdir -p $TMPDIR

echo "Running ICAR visualization on single CPU core"

pwd;hostname;

export PATH="/glade/work/currierw/miniconda3/bin:$PATH"
source activate py3


python icar2movie_crossSection_2d.py > hist.txt


