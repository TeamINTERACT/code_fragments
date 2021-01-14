"""
Usage: python schedule_jobs.py CITY WAVE OUTPUT_DIR [-oie]

Author: Kole Phillips

Script to automate the creation of the Table of Power on Cedar using the Slurm scheduling system

Arguments:
  CITY: The name of the city the table is to represent
  WAVE: The wave number of the study to represent in the table
  OUTPUT_DIR: Location the output files should be stored in

Options:
  -o: Ignore and overwrite the existing table of power csv
  -i: Create a separate job for each individual participant in the study
  -e: Use Ethica data instead of Sensedoc data to create the table of power
"""
import interact_tools as it
import time
from sys import argv
from os.path import isfile, isdir
import subprocess
import os


if __name__ == "__main__":
    city, wave = it.get_command_args("top_generation.py")
    output_dir = argv[3]
    if not isdir(output_dir):
        print("Could not locate directory: " + output_dir)
    if '-i' in argv:
        participants = it.iid_list(city, wave).interact_id.tolist()
        ranges = []
        for p in participants:
            out_fname = output_dir + '/' + str(p) + '_table_of_power_' + it.get_last_commit_date() + '.csv'        
            if '-o' not in argv and isfile(out_fname):                                                    
                continue
            ranges.append((int(p), int(p) + 1))
    else:
        step = 100000
        lowval = it.cities[city] * 100000000 + 1000000
        highval = lowval + int(wave) * 1000000
        ranges = zip(range(lowval, highval, step), range(lowval + step, highval + step, step))

    if len(argv) < 4:
        print("Usage: python top_generation.py SITE_NAME WAVE_NUMBER OUTPUT_DIR [OPTIONS]")
        exit()
    
    aborted = []
    for p, p2 in ranges:
        print((p, p2))
        out_fname = output_dir + '/' + str(p) + '_table_of_power_' + it.get_last_commit_date() + '.csv'
        if '-o' not in argv and isfile(out_fname):
            print("Participant has already been processed.\n")
            continue
        retry = 0
        while retry < 4:
            try:
                f = open("schedule_job_" + str(p) + ".sh", "w")
                f.write("#!/bin/bash\n")
                f.write("#SBATCH --account=def-dfuller\n")
                if '-i' in argv:
                    f.write("#SBATCH --time=04:00:00\n")
                elif '-e' in argv:
                    f.write("#SBATCH --time=10:00:00\n")
                else:
                    f.write("#SBATCH --time=48:00:00\n")
                f.write("#SBATCH --mem=16G\n")
                f.write("export SLURM_ACCOUNT=def-dfuller\n")
                f.write("export SBATCH_ACCOUNT=$SLURM_ACCOUNT\n")
                f.write("export SALLOC_ACCOUNT=$SLURM_ACCOUNT\n")
                f.write("python top_generation.py ")
                f.write(city + " " + str(wave) + " " + output_dir + " " + str(p) + " " + str(p2) + " -s")
                if '-e' in argv:
                    f.write(' -e')
                if '-o' in argv:
                    f.write(' -o')
                f.write('\n')
                f.close()
                while subprocess.run(["sbatch", "schedule_job_" + str(p) + ".sh"]).returncode:
                    print('Retrying')
                    time.sleep(5)
                time.sleep(5)
                os.remove("schedule_job_" + str(p) + ".sh")
                retry = 4
            except:
                retry = retry + 1
                if retry > 3:
                    print("Aborting.")
                    aborted.append(p)
                else:
                    time.sleep(10)
                    print("Failed to submit job. Retry #" + str(retry))

    print("Aborted: " + str(aborted))

