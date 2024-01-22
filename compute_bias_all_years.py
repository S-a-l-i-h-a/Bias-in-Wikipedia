import getopt
import os
import pickle
import sys
from decimal import Decimal
import numpy as np
import time
import logging
import matplotlib.pyplot as plt
import scipy
from scipy.stats import linregress
from sklearn.metrics.pairwise import cosine_similarity
from preprocess_and_train import SaveIterationsCallback
from matplotlib import rc
from evaluate_quality import evaluate_quality
from significance_test_moment import significance_test_moment

"""
    This class contains functions for computing biases and creating plots to visualize the results. 
    
    Author: Katja Schmahl
    
    Based on: 
    https://github.com/ruhulsbu/StereotypicalGenderAssociationsInLanguage
    Jones, Jason & Amin, Mohammad & Kim, Jessica & Skiena, Steven. (2020). 
    Stereotypical Gender Associations in Language Have Decreased Over Time. 10.15195/v7.a1. 

    This code is licensed using a [Creative Commons License](http://creativecommons.org/licenses/by/4.0/). 
    This means you can use, edit and distribute it as long as you give appropriate credit and 
    indicate if changes were made.
"""

logger = logging.getLogger(__name__)
timestamp = time.strftime('%Y%m%d-%H%M')


def main(args):
    if not os.path.exists('logs/compute-bias-all-years'):
        os.makedirs('logs/compute-bias-all-years')
    log_fname = f'logs/compute-bias-all-years/{timestamp}.log'
    logging.basicConfig(filename=log_fname,
                        level=logging.INFO,
                        filemode='w',
                        format='%(asctime)s %(levelname)-8s %(message)s')
    global logger
    logger = logging.getLogger(__name__)

    try:
        opts, args = getopt.getopt(args, "hy:i:b:fs", ["help", "years=", "iterations=", "boxplots=", "filter", "skip-significance"])
    except getopt.GetoptError:
        print('Argument error')
        print('Use compute_bias_all_years.py -y <years> -i <iterations> -b <boxplot-years> [-f -s]')
        print('Use python3 compute_bias -h for more information')
        sys.exit(2)

    if '-h' in args:
        print('Use compute_bias_all_years.py -y <years> -i <iterations> -b <boxplot-years> [-f -s]')
        print('Arguments (all optional) are:')
        print('-y or --years, the years to compute the biases for, these should have been preprocessed and trained already')
        print('-i or --iterations, the amount of iterations the model you want to use has')
        print('-b or --boxplots, the years for which you want to compute random biases for and show in the graph')
        print('Options are:')
        print('-f or --filter if you want to use only articles added in last two years')
        print('-s or --skip-significance if you want to skip computing the significance in comparison to random slopes')
        sys.exit(1)

    else:
        years, filtered, iterations, boxplotyears, skip_significance = None, False, 2, None, False
        for opt, val in opts:
            if opt in ('-y', '--years'):
                try:
                    years = [int(a.strip()) for a in val.split(',')]
                except:
                    print('Exception occured, use comma separated list of years to use')
                    sys.exit(1)
            elif opt in ('-f', '--filter'):
                filtered = True
            elif opt in ('-b', '--boxplots'):
                try:
                    boxplotyears = [int(a.strip()) for a in val.split(',')]
                except:
                    print('Exception occured, use comma separated list of boxplot years to use')
                    sys.exit(1)
            elif opt in ('-i', '--iterations'):
                iterations = int(val)
            elif opt in ('-s', '--skip-significance'):
                skip_significance = True

    compute_bias_all_years(years=years, filtered=filtered, boxplotyears=boxplotyears, iterations=iterations,
                           skip_significance=skip_significance)


def compute_bias_all_years(years=None, filtered=False, iterations=None, boxplotyears=None, skip_significance=False):
    cbal = ComputeBiasAllYears(years=years, filtered=filtered, skip_significance=skip_significance,
                               boxplotyears=boxplotyears, iterations=iterations)
    cbal.full_bias_computations()

class ComputeBiasAllYears:
    # Default years to use
    # years = [2006, 2008, 2009, 2010, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
    # filter_years = [2008, 2010, 2016, 2017, 2018, 2019, 2020]
    # boxplot_years = [2006, 2010, 2015, 2020]
    # filter_boxplot_years = [2008, 2016, 2020]
    years = [2010,2017,2023,2024]
    filter_years = [2010,2017,2023,2024]
    boxplot_years = [2010,2017,2023,2024]
    filter_boxplot_years = [2010,2017,2023,2024]
    # Domains that are used for bias calculations, words are specified in data/weat.txt
    domains = ["WEAT_Topic_Female", "WEAT_Topic_Male", "WEAT_Topic_Family",
               "WEAT_Topic_Career", "WEAT_Topic_Science", "WEAT_Topic_Arts"]
    color = ["red", "green", "blue", "magenta", "brown", "cyan"]
    marker = ['o', 's', 'p', 'd', '>', '<']

    def __init__(self, filtered=False, boxplotyears=None, years=None, iterations=5, skip_significance=False):
        print(sorted(self.years)[-1])
        self.filtered = filtered
        self.skip_significance = skip_significance
        self.iterations = iterations
        self.weat_file_path = "data/weat.txt"
        if filtered:
            self.embedding_file_path = f"data/models/filtered/iter{iterations}/"
            self.word_file_path = f"data/models/filtered/iter{iterations}/{sorted(self.years)[-1]}_filtered_iter{iterations}-vocab.pkl"
            self.years = self.filter_years
            self.boxplot_years = self.filter_boxplot_years
        else:
            self.embedding_file_path = f"data/models/regular/iter{iterations}/"
            self.word_file_path = f"data/models/regular/iter{iterations}/{sorted(self.years)[-1]}_iter{iterations}-vocab.pkl"

        if years is not None:
            self.years = years
        if boxplotyears is not None:
            self.boxplot_years = boxplotyears

        logger.info(f"Computing biases using years {self.years}, filter={filtered}, iterations={iterations},"
                    f"boxplot_years={self.boxplot_years}, skip_significance={skip_significance}")

        word_dict = pickle.load(open(self.word_file_path, "rb"))
        self.word_list = list(word_dict.keys())
        self.word_dic = dict({(x, i) for (i, x) in enumerate(self.word_list)})
        self.word2vec_pkl = {}
        self.word2vec_npy = {}

        for year in self.years:
            if filtered:
                word_file_name = f"{str(year)}_filtered_iter{iterations}-vocab.pkl"
                vec_file_name = f"{year}_filtered_iter{iterations}.vectors.wv.npy"
            else:
                word_file_name = f"{str(year)}_iter{iterations}-vocab.pkl"
                vec_file_name = f"{str(year)}_iter{iterations}.vectors.wv.npy"

            try:
                word_list = pickle.load(open(self.embedding_file_path + word_file_name, "rb"))
                word_vec = np.load(self.embedding_file_path + vec_file_name)
                self.word2vec_pkl[str(year)] = word_list
                self.word2vec_npy[str(year)] = word_vec
                logger.info(f"Loaded word list from {word_file_name} and vectors from {vec_file_name}")
            except:
                print('An error occured, see logs for more information')
                logger.warning(f'Problem opening model for {year} with {iterations} iterations, filtered={filtered}')
                logger.warning(f'Word file path used: {self.embedding_file_path + word_file_name}')
                logger.warning(f'Vector file path used: {self.embedding_file_path + vec_file_name}')

    def full_bias_computations(self):
        self.load_weat_words()
        self.get_overall_bias_all_categories()
        self.plot_all_categories()
        if not self.skip_significance:
            self.significance_test_all_years()
        evaluate_quality(self.years, self.iterations, self.filtered)

    def load_weat_words(self, female_topic="WEAT_Topic_Female", male_topic="WEAT_Topic_Male"):
        file_read = open(self.weat_file_path, "r")
        topic_dict = {}

        for line in file_read:
            data = line.strip().split(",")
            current_topic = data[0]

            if current_topic in self.domains:
                topic_dict[current_topic] = [x.lower() for x in data[1:]]

        self.female_domain = [female_topic] + topic_dict[female_topic]
        self.male_domain = [male_topic] + topic_dict[male_topic]

        del topic_dict[female_topic]
        del topic_dict[male_topic]
        self.domain_dict = topic_dict
        logger.info(f"The weat category words are: {self.domain_dict}")

    def average_similarity_word_vs_domain(self, word_one, given_list):
        wordsim = []
        for year in self.years:
            word_list = self.word2vec_pkl[str(year)]
            word_dic = dict({(x, i) for (i, x) in enumerate(word_list)})
            word_vec = self.word2vec_npy[str(year)]

            similarity = []
            for word_two in given_list:
                try:
                    vec_one = np.array(word_vec[word_dic[word_one]])
                    vec_two = np.array(word_vec[word_dic[word_two]])
                except:
                    return RuntimeWarning

                sim = cosine_similarity([vec_one], [vec_two])
                similarity.append(sim[0][0])

            wordsim.append(np.average(similarity))

        return wordsim

    def get_overall_bias_all_categories(self):
        self.biases = []
        d = 0
        for domain in self.domains[2:]:
            bias_per_word = []
            unknown_word = 0
            for word in self.domain_dict[domain]:
                try:
                    female_assoc = self.average_similarity_word_vs_domain(word, self.female_domain[1:])
                    male_assoc = self.average_similarity_word_vs_domain(word, self.male_domain[1:])
                    biases = np.subtract(male_assoc, female_assoc)
                    bias_per_word.append(biases)
                except:
                    logger.warning(f"Unknown word: {word}")
                    unknown_word += 1
                    if unknown_word > 1:
                        return False

            bias_per_domain = [
                np.average([bias_per_word[i][y] for i in range(0, len(self.domain_dict[domain]) - unknown_word)]) for y
                in range(0, len(self.years))]

            self.biases.append(bias_per_domain)
            d += 1
        return True

    def plot_all_categories(self):
        weatset = ["Family", "Career", "Science", "Arts"]
        bias_scores = []
        regression_params = []

        female_topic = self.female_domain[0]
        male_topic = self.male_domain[0]

        i = 0
        plt.figure()
        for subject in weatset:
            slope, intercept, r_value, p_value, std_err = linregress(self.years, self.biases[i])
            regression_params.append([female_topic + "_" + male_topic, subject,
                                      slope, intercept, r_value, p_value, std_err])
            bias_scores.append([female_topic + "_" + male_topic, subject] + self.biases[i])
            regress_bias = slope * np.array(self.years) + intercept
            plt.plot(self.years, regress_bias, linewidth=3, color=self.color[i], linestyle='dashed', alpha=0.6)
            # plt.plot(self.years, self.biases[i], color=self.color[i], marker=self.marker[i], ms=6, linestyle='-',
            #          linewidth=3, alpha=0.8)
            plt.plot(self.years, self.biases[i], label=subject, marker=self.marker[i], linewidth=2, alpha=0.9, mfc=self.color[i],
                     ms=6, mew=0.4, mec='black', color=self.color[i])
            logger.info(f"slope of {subject} is {slope}")
            i += 1

        if self.filtered:
            plt.title("Gender bias over time on Wikipedia \npages added in the last 2 years", fontsize=18)
        else:
            plt.title(f"Gender bias over time on Wikipedia pages", fontsize=18)
        plt.ylabel(r"Female bias $\leftarrow$ Neutral $\rightarrow$ Male bias     ", fontsize=14)


        # Add boxplots to show distribution of random categories
        bxplt = []
        for year in self.boxplot_years:
            try:
                if self.filtered:
                    file_name = f'data/random-biases/filtered/iter{self.iterations}/{year}-bias-scores_filtered_iter{self.iterations}.pkl'

                else:
                    file_name = f'data/random-biases/regular/iter{self.iterations}/{year}-bias-scores_iter{self.iterations}.pkl'
                logger.info(f'Trying to open {file_name}')
                biases = pickle.load(open(file_name, 'rb'))
            except:
                logger.info(f'No existing random biases found for {year}, computing them')
                biases = significance_test_moment(year, filtered=self.filtered, iterations=self.iterations)
            new_biases = []
            for bias in biases:
                if len(bias) > 0:
                    new_biases.append(bias[0])
            bxplt.append(new_biases)
        whis_props = dict(color="black", alpha=0.4)
        box_props = dict(color="black", alpha=0.8)
        plt.boxplot(bxplt, whis=(5, 95), positions=self.boxplot_years, showfliers=False, capprops=whis_props,
                    whiskerprops=whis_props, boxprops=box_props)

        xaxis = [i for i in range(self.years[0], self.years[len(self.years) - 1] + 1, 1)]
        xtick = [i for i in range(self.years[0], self.years[len(self.years) - 1] + 1, 1)]
        plt.xticks(xaxis, xtick, rotation=45, fontsize=14)

        yaxis = [i/100 for i in range(-8, 9, 2)]
        ytick = [i/100 for i in range(-8, 9, 2)]
        plt.yticks(yaxis, ytick, rotation=45, fontsize=14)
        plt.legend(ncol=2, fontsize=14)

        timestamp = time.strftime('%Y%m%d-%H%M')
        if self.filtered:
            directory = f'results/filtered/iter{self.iterations}'
        else:
            directory = f'results/regular/iter{self.iterations}'
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_name = f'{directory}/biases-plot{"_filtered-" if self.filtered else ""}-{timestamp}.pdf'
        logger.info(f'Saved as {file_name}')
        plt.tight_layout(pad=0.15)
        plt.savefig(file_name, format='pdf')

    def randomize_weat_words(self):
        for domain in self.domains[2:]:
            data_list = []
            for k in range(len(self.domain_dict[domain])):
                randind = np.random.randint(0, len(self.word_list))
                data_list.append(self.word_list[randind])
            self.domain_dict[domain] = data_list

    def significance_test_all_years(self):
        logger.info('Significance testing using random slopes')
        slopes = []
        biases = []
        k = 0
        try:
            file_name_slopes = f'data/random-slopes/random-slopes{"_filtered" if self.filtered else ""}_iter{self.iterations}.pkl'
            file_name_biases = f'data/random-slopes/random-biases{"_filtered" if self.filtered else ""}_iter{self.iterations}.pkl'
            logger.info(f'Loading {file_name_slopes}')
            logger.info(f'Loading {file_name_biases}')
            file = open(file_name_slopes, 'rb')
            file2 = open(file_name_biases, 'rb')
            slopes = pickle.load(file)
            biases = pickle.load(file2)
        except:
            logger.info("Computing 1000 random slopes")
            self.domain_dicts = []
            while k < 1000:
                if k % 50 == 0:
                    print(f"Computed {k} random slopes")
                    logger.info(f"Computed {k} random slopes")
                self.domains = ["WEAT_Topic_Female", "WEAT_Topic_Male", "WEAT_Topic_Family"]
                self.randomize_weat_words()
                success = self.get_overall_bias_all_categories()
                if k % 50 == 0:
                    logger.info(self.biases)
                if success:
                    biases.append(self.biases)
                    self.domain_dicts.append(self.domain_dict)
                    slope, intercept, r_value, p_value, std_err = linregress(self.years, self.biases[0])
                    slopes.append(slope)
                    k += 1

            file_name_slopes = f'data/random-slopes/random-slopes{"_filtered" if self.filtered else ""}_iter{self.iterations}.pkl'
            file_name_biases = f'data/random-slopes/random-biases{"_filtered" if self.filtered else ""}_iter{self.iterations}.pkl'
            file_name_words = f'data/random-slopes/random-words{"_filtered" if self.filtered else ""}_iter{self.iterations}.pkl'
            logger.info("Done, saving")
            file = open(file_name_slopes, 'wb')
            file2 = open(file_name_biases, 'wb')
            file3 = open(file_name_words, 'wb')
            pickle.dump(slopes, file)
            pickle.dump(biases, file2)
            pickle.dump(self.domain_dicts, file3)

        plt.figure(dpi=400)
        xmin, xmax = -0.005, 0.005
        x = np.linspace(xmin, xmax, 11)
        plt.hist(slopes, density=True, bins=50)
        plt.xticks(x, labels=range(-5, 6, 1), fontsize=16)

        mu = np.average(slopes)
        std = np.std(slopes)
        logger.info(f"Average of the random slopes is {mu}")
        logger.info(f"Standard deviation of the random is {std}")

        if self.filtered:
            plt.title(
                f"Distribution of bias slopes in new articles", fontsize=20)
        else:
            plt.title(
                f"Distribution of bias slopes in all articles", fontsize=20)

        plt.ylabel('Probability density', fontsize=16)
        plt.xlabel(r"Becoming more female $\leftarrow  \rightarrow$ Becoming more male  " + '\n' + r"Slope ($\cdot10^{-3}$)", fontsize=16)

        slope_mean = np.average(slopes)
        slope_std = np.std(slopes)
        logger.info(f"Slope Mean/STD: {slope_mean}, {slope_std}")
        self.domains = ["WEAT_Topic_Female", "WEAT_Topic_Male", "WEAT_Topic_Family",
                        "WEAT_Topic_Career", "WEAT_Topic_Science", "WEAT_Topic_Arts"]
        self.load_weat_words()
        self.get_overall_bias_all_categories()
        i = 0
        domain_slopes = {}
        for domain in self.domains[2:]:
            logger.info("---------------------------------------------------------------")
            logger.info("P-values for " + domain)
            slope, intercept, r_value, p_value, std_err = linregress(self.years, self.biases[i])
            logger.info(f"P value for slope of {domain} against slope of 0: {p_value}")
            slope_np = np.array(slopes)
            higher_count = np.sum(abs(slope_np) >= abs(slope))
            domain_slopes[domain] = slope
            probability_higher = higher_count / 1000
            logger.info(f'P-value against random words is {probability_higher}')
            logger.info(f"Slope is {str(slope)}")
            plt.axvline(slope, linewidth=2.,
                        color=self.color[i], label=domain[11:])
            i += 1
            logger.info("---------------------------------------------------------------")

        if self.filtered:
            result_directory = f'results/filtered/iter{self.iterations}'
        else:
            result_directory = f'results/regular/iter{self.iterations}'

        if not os.path.exists(result_directory):
            os.makedirs(result_directory)

        plt.yticks(ticks=range(0, 900, 100), labels=[str(i) for i in range(0, 900, 100)], fontsize=16)
        plt.tight_layout(pad=0.15)
        plt.savefig(
            f'{result_directory}/histogram-no-legend{"-filtered" if self.filtered else ""}-{timestamp}.pdf',
            bbox_inches="tight", format='pdf')
        plt.legend(loc='upper right', ncol=1, borderaxespad=0., fontsize=16)
        plt.savefig(
            f'{result_directory}/histogram-{"filtered-" if self.filtered else ""}{timestamp}.pdf',
            bbox_inches="tight", format='pdf')
        plt.close()
        return slopes, domain_slopes


if __name__ == "__main__":
    # calling main function
    main(sys.argv[1:])
