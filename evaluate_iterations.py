import getopt
import logging
import os
import pickle
import sys
import time
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.word2vec import LineSentence
import matplotlib.pyplot as plt
from gensim.test.utils import datapath
import numpy as np
from scipy.stats import linregress
from sklearn.metrics.pairwise import cosine_similarity
from preprocess_and_train import save_model_in_stereotype_format, SaveIterationsCallback

""" This code was created by Katja Schmahl. This code is licensed using a [Creative Commons License](http://creativecommons.org/licenses/by/4.0/).
This means you can use, edit and distribute it as long as you give appropriate credit and
indicate if changes were made."""

logger = logging.getLogger(__name__)
timestamp = time.strftime('%Y%m%d-%H%M')

domains = ["Family", "Career", "Science", "Arts"]
color = ["red", "green", "blue", "magenta"]
marker = ['o', 's', 'p', 'd', '>', '<']

def main(args):
    if not os.path.exists('logs/evaluate-iterations'):
        os.makedirs('logs/evaluate-iterations')
    log_fname = f"logs/evaluate-iterations/{timestamp}.log"
    logging.basicConfig(filename=log_fname,
                        level=logging.INFO,
                        filemode='w',
                        format='%(asctime)s %(levelname)-8s %(message)s')
    global logger
    logger = logging.getLogger(__name__)

    try:
        opts, args = getopt.getopt(args, "hy:i:l:f", ["years=", "iterations=", "iteration_list=", "filter"])
    except getopt.GetoptError:
        print('Argument error')
        sys.exit(2)

    if '-h' in args:
        print('Use evaluate_iterations.py -y <years> -i <amount-iteartions> [-f]')
        print('Or evaluate_iterations -y <years> -l <iteration-list> [-f]')
        print('Options are ')
        print('-y or --years: the comma separated list of years to use')
        print('-i or --iterations: the amount of iterations to use, every iteration up to this will be evaluated')
        print('-l or --iteration_list: the comma separated list of specific iterations to evaluate')
        sys.exit(1)

    else:
        years, iterations, iteration_list, filtered = [2006,2008,2009,2010,2014,2015,2016,2017,2018,2019,2020], 20, [1, 5, 10, 20], False
        for opt, val in opts:
            if opt in ('-y', '--years'):
                years = [int(i.strip()) for i in list(val.split(','))]
            if opt in ('-i', '--iterations'):
                iterations = int(val)
            if opt in ('-l', '--iteration_list'):
                iteration_list = [int(i.strip()) for i in list(val.split(','))]
            elif opt in ('-f', '--filter'):
                filtered = True

        if iteration_list is None:
            iteration_list = range(1, iterations+1)

        print(filtered)

        if years is not None:
            evaluate_quality_iterations(years, iterations, iteration_list, filtered)
        else:
            print("Missing year arguments")
            sys.exit(1)


def evaluate_quality_iterations(years, iterations, iteration_list, filtered=False):
    logger.info(f'Evaluating years: {years}')
    logger.info(f'Evaluating iterations: {iteration_list}')
    changes = []
    biases_per_iter = []
    for year in years:
        google_scores, pearson_scores, spearman_scores = get_quality_scores(year, iterations, iteration_list, filtered)
        logger.info(google_scores)

        plt.figure(figsize=(8, 6))
        plt.plot(list(pearson_scores.keys()), list(pearson_scores.values()), linestyle='-', linewidth=3, color='blue',
                 label='SIM353 pearson quality scores')
        plt.plot(list(spearman_scores.keys()), list(spearman_scores.values()), linestyle='-', linewidth=3,
                 color='green', label='SIM353 spearman quality scores')
        if not os.path.exists('results/quality'):
            os.makedirs('results/quality')
        plt.savefig(f'results/quality/{year}-iterations')

        iterations = BiasIterations(year, iterations, iteration_list=iteration_list, filtered=filtered)
        changes.append(iterations.evaluate_bias())
        biases_per_iter.append(iterations.get_overall_bias_all_categories())

    plot_slopes(biases_per_iter, years, iteration_list, filtered)
    all_values = []
    for change_list in changes:
        for value in change_list:
            all_values.append(value)

    lists = {}
    print(lists)
    for it in iteration_list:
        lists[it] = []

    for year in years:
        google_scores, pearson_scores, spearman_scores = get_quality_scores(year, 1, iteration_list, filtered)
        for it in iteration_list:
            lists[it].append(pearson_scores[it])

    print(lists)


def plot_slopes(bias_per_iter, years, iteration_list, filtered):
    slopes = {subject: [] for subject in domains}
    print(bias_per_iter)
    plt.figure()
    s = 0
    for subject in domains:
        i = 0
        for iter in iteration_list:
            slopes[subject].append(linregress(years, [bi[s][i] for bi in bias_per_iter])[0])
            i += 1
        plt.plot(iteration_list, slopes[subject], marker=marker[s], label=subject, linewidth=2, alpha=0.8, mfc=color[s],
                 ms=6, mec='black', mew=1.25, color=color[s])
        s += 1

    xticks = [i for i in range(0, max(iteration_list) + 1)]
    xlabels = [str(i) for i in range(0, max(iteration_list) + 1)]
    yticks = [i * 0.001 for i in range(-3, 4)]
    ylabels = [i for i in range(-3, 4)]
    plt.xticks(ticks=xticks, labels=xlabels)
    plt.yticks(ticks=yticks, labels=ylabels)
    plt.title('Bias slopes over iterations')
    plt.xlabel('Amount of iterations')
    plt.ylabel(r'Slope of biases ($\cdot10^{-3}$)')
    plt.legend()
    plt.savefig(
        f'results/iterations/slopes-over-iterations-{"filtered" if filtered else ""}{timestamp}')


def get_quality_scores(year, iterations, iteration_list, filter):
    google_scores = {}
    pearson_scores = {}
    spearman_scores = {}

    for i in iteration_list:
        if not filter:
            print(f'{i} - {year}')
            try:
                model = Word2Vec.load(f'data/models/regular/iter{i}/{year}_iter{i}')
            except:
                print('except')
                return google_scores, pearson_scores, spearman_scores
        else:
            model = Word2Vec.load(f'data/models/filtered/iter{i}/{year}_filtered_iter{i}')

        pearson, spearman, oov = model.wv.evaluate_word_pairs(datapath('wordsim353.tsv'))
        pearson_scores[i] = pearson[0]
        spearman_scores[i] = spearman.correlation
        logger.info("SPEARMAN SCORES: ")
        logger.info(f"{spearman_scores}")
        logger.info("PEARSON SCORES:")
        logger.info(f"{pearson_scores}")
    return google_scores, pearson_scores, spearman_scores


class BiasIterations:
    domains = ["WEAT_Topic_Female", "WEAT_Topic_Male", "WEAT_Topic_Family",
               "WEAT_Topic_Career", "WEAT_Topic_Science", "WEAT_Topic_Arts"]
    color = ["red", "green", "blue", "magenta", "brown", "cyan"]
    marker = ['o', 's', 'p', 'd', '>', '<']

    def __init__(self, year, iterations=10, iteration_list=None, filtered=False):
        self.weat_file_path = "data/weat.txt"
        if iteration_list is None:
            self.iteration_list = range(1, iterations+1)
        else:
            self.iteration_list = iteration_list
        self.word_file_path = f"iter{self.iteration_list[0]}/{year}_iter{self.iteration_list[0]}-vocab.pkl"
        self.year = year
        self.filtered = filtered
        if self.filtered:
            self.embedding_file_path = "data/models/filtered/"
        else:
            self.embedding_file_path = "data/models/regular/"
        self.iterations = iterations
        word_dict = pickle.load(open(self.embedding_file_path + self.word_file_path, "rb"))
        self.word_list = list(word_dict.keys())
        self.word_dic = dict({(x, i) for (i, x) in enumerate(self.word_list)})
        self.word2vec_pkl = {}
        self.word2vec_npy = {}

        for iter in iteration_list:
            print(f'bias for {year} - {iter}')
            word_file_name = f"iter{iter}/{str(year)}{'_filtered' if filtered else ''}_iter{iter}-vocab.pkl"
            vec_file_name = f"iter{iter}/{str(year)}{'_filtered' if filtered else ''}_iter{iter}.vectors.wv.npy"
            word_list = pickle.load(open(self.embedding_file_path + word_file_name, "rb"))
            logger.info(f"Loaded {word_file_name}")
            word_vec = np.load(self.embedding_file_path + vec_file_name)
            logger.info(f"Loaded vectors from {vec_file_name}")

            self.word2vec_pkl[str(iter)] = word_list
            self.word2vec_npy[str(iter)] = word_vec

    def evaluate_bias(self):
        self.load_weat_words()
        self.get_overall_bias_all_categories()
        return self.plot_all_categories()

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
        for iter in self.iteration_list:
            word_list = self.word2vec_pkl[str(iter)]
            word_dic = dict({(x, i) for (i, x) in enumerate(word_list)})
            word_vec = self.word2vec_npy[str(iter)]

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
                    if unknown_word > 2:
                        return False

            bias_per_domain = [
                np.average([bias_per_word[i][y] for i in range(0, len(self.domain_dict[domain]) - unknown_word)]) for y
                in range(0, len(self.iteration_list))]
            logger.info(f"Bias of {domain} has mean {np.average(bias_per_word)} and std {np.std(bias_per_word)}")
            self.biases.append(bias_per_domain)
            d += 1

        logger.info(f"Done getting biases {self.biases}")
        return self.biases


    def plot_all_categories(self):
        logger.info('Plotting all categories')
        weatset = ["Family", "Career", "Science", "Arts"]
        bias_scores = []
        regression_params = []

        female_topic = self.female_domain[0]
        male_topic = self.male_domain[0]

        i = 0
        plt.figure(figsize=(8, 6))
        changes = []
        for subject in weatset:
            bias_scores.append([female_topic + "_" + male_topic, subject] + self.biases[i])

            plt.plot(self.iteration_list, self.biases[i], color=self.color[i], marker=self.marker[i], ms=6, linestyle='-',
                     linewidth=4, alpha=1)
            changes.append(max(self.biases[i][1:3]) - min(self.biases[i][1:3]))
            plt.plot(self.iteration_list, self.biases[i], label=subject, marker=self.marker[i], linewidth=0.1, mfc=self.color[i],
                     ms=6, mec='black', mew=1.25)
            i += 1

        plt.title(f"Gender bias over iterations on Wikipedia pages from {self.year}")
        plt.ylabel("Male Gender Bias")

        xaxis = range(1, max(self.iteration_list)+1)
        xtick = range(1, max(self.iteration_list)+1)

        plt.xticks(xaxis, xtick, rotation=45)

        plt.legend(bbox_to_anchor=(0., 1.1, 1., .102), loc=4,
                   ncol=4, mode="expand", borderaxespad=0.)

        timestamp = time.strftime('%Y%m%d-%H%M')
        if not os.path.exists('results/iterations/'):
            os.makedirs('results/iterations')
        plt.savefig(
            f'results/iterations/{self.year}-over-iterations-{"filtered" if self.filtered else ""}{timestamp}')
        plt.close()
        return changes

if __name__ == "__main__":
    # calling main function
    main(sys.argv[1:])
