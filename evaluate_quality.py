import getopt
import logging
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.test.utils import datapath

""" This code was created by Katja Schmahl. This code is licensed using a [Creative Commons License](http://creativecommons.org/licenses/by/4.0/). 
This means you can use, edit and distribute it as long as you give appropriate credit and 
indicate if changes were made."""

logger = logging.getLogger(__name__)
timestamp = time.strftime('%Y%m%d-%H%M')


def main(args):
    if not os.path.exists('logs/quality/'):
        os.makedirs('logs/quality')
    log_fname = f"logs/quality/{timestamp}.log"
    logging.basicConfig(filename=log_fname,
                        level=logging.INFO,
                        filemode='w',
                        format='%(asctime)s %(levelname)-8s %(message)s')
    global logger
    logger = logging.getLogger(__name__)

    try:
        opts, args = getopt.getopt(args, "hy:i:f", ["years=", "iterations=", "filter"])
    except getopt.GetoptError:
        print('Argument error')
        sys.exit(2)

    if '-h' in args:
        print('Use evaluate_quality.py -y <years-comma-separated-list> -i <iterations> [-f]')
        sys.exit(1)

    else:
        print('arguments:')
        begin, end, step, years, filtered, iterations = 0, 0, 0, [2008, 2010, 2016, 2017, 2018, 2019, 2020], False, 5
        for opt, val in opts:
            print(opt)
            if opt in ('-b', '--begin'):
                begin = int(val)
            elif opt in ('-e', '--end'):
                end = int(val)
            elif opt in ('-s', '--step'):
                step = int(val)
            elif opt in ('-y', '--years'):
                years = list(val.split(','))
                print(years)
            elif opt in ('-f', '--filter'):
                filtered = True
            elif opt in ('-i', '--iterations'):
                iterations = int(val)

        evaluate_quality(years, int(iterations), filtered)


def evaluate_quality(years, iterations, filtered=False):
    logger.info('Evaluating quality')
    if not os.path.exists('results/quality'):
        os.makedirs('results/quality')
    
    pearson_scores, spearman_scores = get_quality_scores(years, iterations, filtered)
    table = ""
    for key in pearson_scores.keys():
        table = table + ";" + str(pearson_scores[key])
    table = table + '\n'
    for key in spearman_scores.keys():
        table = table + ";" + str(spearman_scores[key])
    logger.info(table)

    plt.figure(figsize=(8, 6))
    plt.title("SIM353 quality scores")
    plt.plot(list(pearson_scores.keys()), list(pearson_scores.values()), linestyle='-', linewidth=3, color='blue', label='Pearson correlation')
    plt.plot(list(spearman_scores.keys()), list(spearman_scores.values()), linestyle='-', linewidth=3, color='green', label='Spearman correlation')
    plt.legend()
    plt.savefig(f'results/quality/quality-{"filtered-" if filtered else ""}-{iterations}allyears-{timestamp}')
    logger.info(f'Average pearson quality score is {np.average(list(pearson_scores.values()))}')
    logger.info(f'Average spearman quality score is {np.average(list(spearman_scores.values()))}')


def get_quality_scores(years, iterations, filtered):
    pearson_scores = {}
    spearman_scores = {}

    for year in years:
        if not filtered:
            model = Word2Vec.load(f'data/models/regular/iter{iterations}/{year}_iter{iterations}')
        else:
            model = Word2Vec.load(f'data/models/filtered/iter{iterations}/{year}_filtered_iter{iterations}')
        pearson, spearman, oov = model.wv.evaluate_word_pairs(datapath('wordsim353.tsv'))
        pearson_scores[year] = pearson[0]
        spearman_scores[year] = spearman.correlation
        logger.info(f'{year}')
        logger.info("SPEARMAN SCORES: ")
        logger.info(f"{spearman_scores}")
        logger.info("PEARSON SCORES:")
        logger.info(f"{pearson_scores}")

    return pearson_scores, spearman_scores


if __name__ == "__main__":
    # calling main function
    main(sys.argv[1:])
