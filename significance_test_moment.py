import getopt
import logging
import os
import pickle
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from compute_gender_bias_wiki import Gender_Bias_Wiki
from plotting_utility import compute_topic_assoc

""" This code was created by Katja Schmahl. This code is licensed using a [Creative Commons License](http://creativecommons.org/licenses/by/4.0/). 
This means you can use, edit and distribute it as long as you give appropriate credit and 
indicate if changes were made."""

logger = logging.getLogger(__name__)


color = ["red", "green", "blue", "magenta", "brown", "cyan"]
timestamp = time.strftime('%Y%m%d-%H%M')


def main(args):
    if not os.path.exists('logs/significance-test-moment'):
        os.makedirs('logs/significance-test-moment')
    log_fname = f'logs/significance-test-moment/{timestamp}'
    logging.basicConfig(filename=log_fname,
                        level=logging.INFO,
                        filemode='w',
                        format='%(asctime)s %(levelname)-8s %(message)s')
    global logger
    logger = logging.getLogger(__name__)

    try:
        opts, args = getopt.getopt(args, "hy:i:f", ["year=", "iterations=", "filter"])
    except getopt.GetoptError:
        print('Argument error')
        print('Use significance_test_moment.py -y <year> [-i <iterations> -f]')
        sys.exit(2)

    if '-h' in args:
        print('Use significance_test_moment.py -y <year> [-i <iterations> -f]')
        sys.exit(1)

    else:
        year, filtered, iterations = None, False, 5
        for opt, val in opts:
            if opt in ('-y', '--year'):
                year = int(val)
            if opt in ('-f', '--filter'):
                filtered = True
            if opt in ('-i', '--iterations'):
                iterations = int(val)

        significance_test_moment(year, filtered, iterations=iterations)


def significance_test_moment(year, filtered=False, iterations=1):
    logger.info(f"Significance test moment for {year} with filter={filtered} and iterations {iterations}")
    domains = ["WEAT_Topic_Female", "WEAT_Topic_Male", "WEAT_Topic_Family"]
    if filtered:
        gender_profile_lang = Gender_Bias_Wiki(domains, str(Gender_Bias_Wiki.BEGIN_YEAR + year), filtered=filtered,
                                               iterations=iterations, postfix=f'_filtered_iter{iterations}')
    else:
        gender_profile_lang = Gender_Bias_Wiki(domains, str(Gender_Bias_Wiki.BEGIN_YEAR + year), filtered=filtered,
                                               iterations=iterations, postfix=f'_iter{iterations}')
    gender_profile_lang.load_embeddings(year, year+1, 2)
    gender_profile_lang.load_weat_words()
    gender_profile_lang.create_data_store_stats()

    female_topic = gender_profile_lang.female_domain[0]
    female_words = gender_profile_lang.female_domain[1:]

    male_topic = gender_profile_lang.male_domain[0]
    male_words = gender_profile_lang.male_domain[1:]

    female_scores = []
    male_scores = []
    bias_scores = []
    try:
        if filtered:
            pickle_file = open(f'data/random-biases/{"filtered" if filtered else "regular"}/iter{iterations}/{year}-bias-scores_filtered_iter{iterations}.pkl', 'rb')
            bias_scores = pickle.load(pickle_file)
        else:
            pickle_file = open(f'data/random-biases/{"filtered" if filtered else "regular"}/iter{iterations}/{year}-bias-scores_iter{iterations}.pkl', 'rb')
            bias_scores = pickle.load(pickle_file)

    except FileNotFoundError:
        for k in range(0, 1000):
            gender_profile_lang.randomize_weat_words()
            if k % 50 == 0:
                logger.info(f"Random biases in year, progress {k}/1000")
            gender_profile_lang.create_data_store_stats()
            female_topic_assoc = compute_topic_assoc(gender_profile_lang, female_topic, female_words, 'Family')
            male_topic_assoc = compute_topic_assoc(gender_profile_lang, male_topic, male_words, 'Family')
            topic_bias = np.subtract(male_topic_assoc, female_topic_assoc)

            male_scores.append(male_topic_assoc)
            female_scores.append(female_topic_assoc)
            bias_scores.append(topic_bias)
    except:
        print("An unexpected error occured when trying to open the pickle files")
        sys.exit(2)

    bias_mean = np.average(bias_scores)
    bias_std = np.std(bias_scores)
    logger.info(f"Bias Mean/STD: {bias_mean}, {bias_std}")

    if filtered:
        directory = f'data/random-biases/filtered/iter{iterations}'
    else:
        directory = f'data/random-biases/regular/iter{iterations}'

    if not os.path.exists(directory):
        os.makedirs(directory)
    pickle_file = open(f'{directory}/{year}-bias-scores{"_filtered" if filtered else ""}_iter{iterations}.pkl', 'wb')
    pickle.dump(bias_scores, pickle_file)
    plt.figure(figsize=(8, 6))
    plt.hist([bias_scores[i][0] for i in range(0, 1000)], bins=50)

    plt.ylabel("Count of Simulated Values")
    plt.title(f"Histogram of Values for Random Word Categories in {year} from articles added in last two years")

    timestamp = time.strftime('%Y%m%d-%H%M')
    dir = f'results/bias-histograms'
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(f'{dir}/{year}-histogram-{timestamp}.png')
    plt.close()
    return bias_scores


if __name__ == "__main__":
    # calling main function
    main(sys.argv[1:])
