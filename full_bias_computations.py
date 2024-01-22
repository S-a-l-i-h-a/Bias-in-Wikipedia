import getopt
import logging
import os
import sys
import time

from compute_bias_all_years import ComputeBiasAllYears as cbal, compute_bias_all_years
from preprocess_and_train import preprocess_and_train_data

# import debugpy
# debugpy.listen(("127.0.0.1", 5678))


""" This code was created by Katja Schmahl. This code is licensed using a [Creative Commons License](http://creativecommons.org/licenses/by/4.0/). 
This means you can use, edit and distribute it as long as you give appropriate credit and 
indicate if changes were made."""


logger = logging.getLogger(__name__)
timestamp = time.strftime('%Y%m%d-%H%M')


def main(args):
    if not os.path.exists('logs/full-bias-computations'):
        os.makedirs('logs/full-bias-computations')
    log_fname = f"logs/full-bias-computations/{timestamp}.log"
    open(log_fname, 'a').close()
    logging.basicConfig(filename=log_fname,
                        level=logging.INFO,
                        filemode='w',
                        format='%(asctime)s %(levelname)-8s %(message)s')
    global logger
    logger = logging.getLogger(__name__)

    try:
        opts, args = getopt.getopt(args, "hsi:p", ["help", "save-iterations", "iterations=", "preprocess"])
    except getopt.GetoptError:
        print('Use full_bias_computations.py [-s -i <iterations> -p]')
        sys.exit(2)

    if '-h' in args:
        print('Use full_bias_computations.py [-s -i <iterations> -p]')
        print('This will preprocess and train from all dumps and compute all biases and significance scores')
        print('WARNING: this will take really long, to do the steps more gradually and controlled, ')
        sys.exit(1)

    else:
        save_iter, iterations, preprocess = False, 5, True
        for opt, arg in opts:
            if opt in ('-s', '--save-iterations'):
                save_iter = True
            elif opt in ('-i', '--iterations'):
                iterations = int(arg)
            elif opt in ('-p', '--preprocess'):
                preprocess = False

        full_bias_computations(save_all_iterations=save_iter, iterations=iterations, preprocess=preprocess, filter_years=[])


def full_bias_computations(years=cbal.years, filter_years=cbal.filter_years, preprocess=True, iterations=5,
                           save_all_iterations=True):
    for year in years:
        if preprocess:
            logger.info(f'Preprocessing and training the regular years from compute_bias_all_years.py: {cbal.years}')
            if not os.path.exists(f'data/raw/{year}.xml.bz2'):
                print('A dump file is missing, make sure you have a dump for every year '
                      'in compute_bias_all_years.py with path data/raw/<year>.xml.bz2')
                sys.exit(1)
            preprocess_and_train_data(inp=f'data/raw/{year}.xml.bz2', year=year, load=False, filtered=True, iterations=iterations,
                                      seed=0, save_iterations=save_all_iterations)
        else:
            if not os.path.exists(f'data/models/regular/iter{iterations}/{year}_iter{iterations}'):
                if not os.path.exists(f'data/corpus/corpus_{year}.txt'):
                    print(f'A corpus file is missing for year {year}, make sure you have a corpus for every year '
                            f'in compute_bias_all_years.py with path data/corpus/corpus_<year>.txt'
                            f'or dont use -p option to create these')
                sys.exit(1)
            preprocess_and_train_data(inp=None, year=year, load=True, filtered=True,
                                iterations=iterations,
                                seed=0, save_iterations=save_all_iterations)
    for year in filter_years:
    	if preprocess:
            logger.info(f'Preprocessing and training the regular years from compute_bias_all_years.py: {cbal.years}')
            if not os.path.exists(f'data/raw/{year}.xml.bz2'):
                print('A dump file is missing, make sure you have a dump for every year '
                      'in compute_bias_all_years.py with path data/raw/<year>.xml.bz2')
                sys.exit(1)
            preprocess_and_train_data(inp=f'data/raw/{year}_filtered.xml.bz2', year=year, load=False, filtered=True,
                                   	  iterations=iterations, seed=0, save_iterations=save_all_iterations)

    else:
        if not os.path.exists(f'data/models/filtered/iter{iterations}/{year}_filtered_iter{iterations}'):
            if not os.path.exists(f'data/corpus/corpus_{year}_filtered.xml.bz2'):
                print('A corpus file is missing, make sure you have a corpus for every year '
                        'in compute_bias_all_years.py with path data/corpus/corpus_<year>_filtered.txt'
                        'or dont use -p option to create these')
                sys.exit(1)
            preprocess_and_train_data(inp=None, year=year, load=False, filtered=True,
                                    iterations=iterations, seed=0, save_iterations=save_all_iterations)

    compute_bias_all_years(years=years, filtered=False, iterations=iterations, skip_significance=False)
    compute_bias_all_years(years=years, filtered=True, iterations=iterations, skip_significance=False)


if __name__ == "__main__":
    main(sys.argv[1:])
