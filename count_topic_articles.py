import getopt
import logging
import os
import pickle
import sys
import time
import matplotlib.pyplot as plt

""" This code was created by Katja Schmahl. This code is licensed using a [Creative Commons License](http://creativecommons.org/licenses/by/4.0/). 
This means you can use, edit and distribute it as long as you give appropriate credit and 
indicate if changes were made. """

logger = logging.getLogger(__name__)
timestamp = time.strftime('%Y%m%d-%H%M')
weat_file_path = 'data/weat.txt'
DEFAULT_YEARS = [2024]


def main(args):
    if not os.path.exists('logs/counts'):
        os.makedirs('logs/counts')
    log_fname = f'logs/counts/{timestamp}.log'
    logging.basicConfig(filename=log_fname,
                        level=logging.INFO,
                        filemode='w',
                        format='%(asctime)s %(levelname)-8s %(message)s')
    global logger
    logger = logging.getLogger(__name__)

    try:
        opts, args = getopt.getopt(args, "hy:", ["years="])
    except getopt.GetoptError:
        print('Argument error')
        print('Use count_topic_articles -y <years>')
        sys.exit(2)

    if '-h' in args:
        print('Use count_topic_articles -y <years>')
        sys.exit(1)

    else:
        for opt, val in opts:
            if opt in ('-y', '--years'):
                try:
                    years = [int(a.strip()) for a in val.split(',')]
                    count_topic_articles_multiple_years(years)
                except:
                    print('Exception occured, use comma separated list of years to use')
                    sys.exit(1)

        count_topic_articles_multiple_years()


def count_topic_articles_multiple_years(years=DEFAULT_YEARS):
    try:
        file = open('data/counts/word-counts.pkl', 'rb')
        year_dict = pickle.load(file)
        logger.info("Loaded existing counts")
    except:
        logger.info("No existing counts found")
        year_dict = {}
    overall_counts = []
    topic_counts = []
    for year in years:
        logger.info(f"Looking for {year}")
        if str(year) in year_dict.keys():
            logger.info(f"Using existing counts")
            (overall, topics) = year_dict[str(year)]
        else:
            logger.info(f"Counting from corpus")
            overall, topics = count_topic_articles(year)
            if overall is not None:
                year_dict[str(year)] = (overall, topics)

        if overall is not None:
            overall_counts.append(overall)
            topic_counts.append(topics)

    logger.info(f"Overall counts: {overall_counts}")
    logger.info(f"Topic counts: {topic_counts}")
    plot_normalized_line_chart(years, overall_counts, topic_counts)

    if not os.path.exists('data/counts'):
        os.makedirs('data/counts')
    file = open('data/counts/word-counts.pkl', 'wb')
    pickle.dump(year_dict, file)
    logger.info("Stored the counts in data/counts.pkl")


def count_topic_articles(year):
    logger.info(" --------------------------------------------------------------------------------")
    logger.info(f" Counting from {year}")
    logger.info(" --------------------------------------------------------------------------------")
    txt_file = f"data/corpus/corpus_{year}.txt"
    if not os.path.exists(txt_file):
        print(f'No corpus found for counting from {year}')
        logger.warning(f'No corpus found for {year}')
        return None, None

    file = open(txt_file, "r")

    overall_count = 0
    topic_counts = [{}, {}, {}, {}, {}, {}]

    file_read = open(weat_file_path, "r")
    topic_dict = {}

    domains = ['WEAT_Topic_Career', 'WEAT_Topic_Family', 'WEAT_Topic_Arts', 'WEAT_Topic_Science', 'WEAT_Topic_Male',
                   'WEAT_Topic_Female']
    for line in file_read:
        data = line.strip().split(",")
        current_topic = data[0]

        if current_topic in domains:
            topic_dict[current_topic] = [x.lower() for x in data[1:]]

    for i in range(0, 6):
        for word in topic_dict[domains[i]]:
            topic_counts[i][word] = 0
            topic_counts[i]['overall-words'] = 0
            topic_counts[i]['overall-articles'] = 0

    for line in file:
        if overall_count % 100000 == 0:
            print_counts(overall_count, topic_counts)
        overall_count += 1
        for i in range(0, len(topic_dict)):
            flag = False
            for word in topic_dict[domains[i]]:
                count = line.count(word)
                if count > 0:
                    topic_counts[i]['overall-words'] += count
                    if not flag:
                        topic_counts[i]['overall-articles'] += 1
                        flag = True
                    topic_counts[i][word] += count

    print_counts(overall_count, topic_counts)
    return overall_count, topic_counts


def print_counts(overall_count, topic_counts):
    topics = ["Career", "Family", "Arts", "Science", "Male", "Female"]
    logger.info(f"topics: {topics}")
    logger.info(f"Total articles: {overall_count}")
    logger.info(f"Articles about {topics}: {topic_counts}")
    per_topic_overall = [topic_counts[i]['overall-words'] for i in range(0, 6)]
    per_topic_count = [topic_counts[i]['overall-articles'] for i in range(0, 6)]
    logger.info(f"Overall counts: {per_topic_overall} ")
    logger.info(f"Overall article counts: {per_topic_count} ")


def plot_normalized_line_chart(years, overall_counts, topic_counts):
    color = ["green", "red", "magenta", "blue", "brown", "cyan"]
    topics = ["Career", "Family", "Arts", "Science", "Male", "Female"]
    labels = [str(year) for year in years]
    all_counts = []
    for i in range(0, len(topics)):
        all_counts.append([])
        for j in range(0, len(years)):
            all_counts[i].append(topic_counts[j][i]['overall-articles'] / overall_counts[j] * 100)

    plt.figure(figsize=(6.4, 5.3))
    for i in range(0, len(topics)):
        plt.plot(years, all_counts[i], label=topics[i], color=color[i], marker='o', linestyle='-', linewidth=2)

    plt.ylabel('Percentage of total articles', fontsize=15)
    plt.title('Prevalence of topics', fontsize=20)
    plt.xticks(range(DEFAULT_YEARS[0], DEFAULT_YEARS[len(DEFAULT_YEARS) - 1] + 1, 1), rotation=45, labels=[str(year) for year in range(DEFAULT_YEARS[0], DEFAULT_YEARS[len(DEFAULT_YEARS) - 1] + 1, 1)], fontsize=12)
    plt.yticks(range(0, 105, 10), rotation=45, labels=[str(perc) for perc in range(0, 105, 10)], fontsize=12)

    plt.legend(loc=(0.02, 0.4), fontsize=11.5, markerscale=0.8, ncol=2, columnspacing=0.5)
    # fontsize = 11, ncol = 1, loc = 2, borderaxespad = 0, borderpad = 0.2, edgecolor = 'black'

    # fig.tight_layout()

    timestamp = time.strftime('%Y%m%d-%H%M')
    if not os.path.exists('results/counts'):
        os.makedirs('results/counts')
    plt.savefig(f'results/counts/normalized-counts-{timestamp}')
    plt.show()


if __name__ == "__main__":
    main(args=sys.argv)
