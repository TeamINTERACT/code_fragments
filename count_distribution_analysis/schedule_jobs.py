"""
Usage: python schedule_jobs.py

Author: Kole Phillips

Script to automate the running of the slope_calcs.py script on Cedar using the Slurm scheduling system
"""
import time
from sys import argv
import subprocess
import os


if __name__ == "__main__":
    cycles = [0, 1, 2, 4, 8]
    sets = [('saskatoon', '1', '/home/kdp740/projects/def-dfuller/interact/testing/saskatoon_01_table_of_power_2020-11-14_sd.csv'), ('victoria', '1', '/home/kdp740/projects/def-dfuller/interact/testing/victoria_01_table_of_power_2020-11-14_sd.csv'), ('vancouver', '1', '/home/kdp740/projects/def-dfuller/interact/testing/vancouver_01_table_of_power_2020-11-14_sd.csv'), ('montreal', '1', '/home/kdp740/projects/def-dfuller/interact/testing/montreal_01_table_of_power_2020-11-14_sd.csv'), ('victoria', '2', ''), ('vancouver', '2', '')]
    i = 0
    for cycle in cycles:
        for city, wave, fname in sets:
            print((city, cycle))
            i = i + 1
            f = open("schedule_job_" + str(i) + ".sh", "w")
            f.write("#!/bin/bash\n")
            f.write("#SBATCH --account=def-dfuller\n")
            if '-old' in argv:
                f.write("#SBATCH --time=02:00:00\n")
            else:
                f.write("#SBATCH --time=01:00:00\n")
            if '-f' in argv:
                f.write("#SBATCH --mem=128G\n")
            else:
                f.write("#SBATCH --mem=4G\n")
            f.write("export SLURM_ACCOUNT=def-dfuller\n")
            f.write("export SBATCH_ACCOUNT=$SLURM_ACCOUNT\n")
            f.write("export SALLOC_ACCOUNT=$SLURM_ACCOUNT\n")
            f.write("python slope_calcs.py  ")
            f.write(city + " " + str(wave) + " " + str(cycle))
            if '-f' in argv:
                f.write(' ' + fname + ' ' + '-f')
            f.write('\n')
            f.close()
            while subprocess.run(["sbatch", "schedule_job_" + str(i) + ".sh"]).returncode:
                print('Retrying')
                time.sleep(5)
            time.sleep(5)
            os.remove("schedule_job_" + str(i) + ".sh")

