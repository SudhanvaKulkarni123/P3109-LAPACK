import os
import sys
import time
import logging
from datetime import timedelta


start = time.time()
logging.basicConfig(filename='./timer.txt', format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

benchlist = [sys.argv[1]]
TIMEOUT = int(sys.argv[2])
PRCEISION = int(sys.argv[3])
DIM = int(sys.argv[4])
COND = int(sys.argv[5])

for B in benchlist:
    logging.info(f"Bench --> {B}")
    for epsilon in [4]:
        _start = time.time()
        # Explanation of the following commands:
            # python3 generate-include.py; --> to generate include.json that specifies functions and variables to be considered
            # python3 setup.py {B}; --> to preprocess the benchmark
            # rm -f *.txt; --> to remove output files from previous runs
            # python3 create-search-space.py {B}; --> to create the search space file called search_config.json and the initial config config.json 
            # python3 ../dd2.py {B} search_config.json config.json {TIMEOUT} {epsilon} A --> to run precimonious on the benchmark

        os.system(f"cd {B}/scripts; \
                    python3 gen.py {PRCEISION} {DIM} {COND}; \
                    cp -rf matrix_collection_{PRCEISION} ../tempscripts/; \
                    cd ../../; \
                    cd {B}/run; \
                    python3 generate-include.py; \
                    python3 setup.py {B}; \
                    python3 create-search-space.py {B}; \
                    python3 ../dd2.py {B} search_config.json config.json {TIMEOUT} {epsilon} A {PRCEISION} {DIM} {COND} \
                        ")
        _elapsed = (time.time() - _start)
        logging.info(f"      epsilon --> {epsilon}  time --> {str(timedelta(seconds=_elapsed))}")

elapsed = (time.time() - start)
logging.info(f"TOTAL TIME: {str(timedelta(seconds=elapsed))}")
