import logging
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class Gender_Bias_Wiki:
    BEGIN_YEAR = 0
    """
        This class contains help functions for computing the biases. Source: 
        https://github.com/ruhulsbu/StereotypicalGenderAssociationsInLanguage
        Jones, Jason & Amin, Mohammad & Kim, Jessica & Skiena, Steven. (2020). 
        Stereotypical Gender Associations in Language Have Decreased Over Time. 10.15195/v7.a1. 
        
        Altered by Katja Schmahl to use different file paths and step sizes.
 
        This code is licensed using a [Creative Commons License](http://creativecommons.org/licenses/by/4.0/). 
        This means you can use, edit and distribute it as long as you give appropriate credit and 
        indicate if changes were made.
    """

    def __init__(self, domains, vocab_year='2020', filtered=False, iterations=5, postfix=''):
        logger.info(f'Gender bias wiki with {vocab_year} and {postfix}')
        self.weat_file_path = "data/weat.txt"
        if filtered:
            self.embedding_file_path = f"data/models/filtered/iter{iterations}/"
            self.word_file_path = f"{self.embedding_file_path}{vocab_year}{postfix}-vocab.pkl"
        else:
            self.embedding_file_path = f"data/models/regular/iter{iterations}/"
            self.word_file_path = f"{self.embedding_file_path}{vocab_year}{postfix}-vocab.pkl"
        self.domains = domains
        self.postfix = postfix

    def load_embeddings(self, start, end, step):
        logger.info(f'Loading embeddings for start {start}, end {end} and step {step}')
        self.start = start
        self.end = end
        self.step = step
        word_dict = pickle.load(open(self.word_file_path, "rb"))
        self.word_list = list(word_dict.keys())
        self.word_dic = dict({(x, i) for (i, x) in enumerate(self.word_list)})
        self.word2vec_pkl = {}
        self.word2vec_npy = {}

        BEGIN_YEAR = Gender_Bias_Wiki.BEGIN_YEAR
        for year in range(start, end, step):
            word_file_name = f"{str(BEGIN_YEAR + year)}{self.postfix}-vocab.pkl"
            word_list = pickle.load(open(self.embedding_file_path + word_file_name, "rb"))
            logger.info(f"Loaded {word_file_name}")
            word_dic = dict({(x, i) for (i, x) in enumerate(word_list)})
            vec_file_name = str(BEGIN_YEAR+year) + f"{self.postfix}.vectors.wv.npy"
            word_vec = np.load(self.embedding_file_path + vec_file_name)
            logger.info(f"Loaded vectors from {vec_file_name}")

            self.word2vec_pkl[str(BEGIN_YEAR+year)] = word_list
            self.word2vec_npy[str(BEGIN_YEAR+year)] = word_vec

    def load_weat_words(self, female_topic="WEAT_Topic_Female", male_topic="WEAT_Topic_Male"):

        file_read = open(self.weat_file_path, "r")
        topic_dict = {}

        logger.info("WEAT Dataset Loading")

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

    def randomize_weat_words(self):

        for domain in self.domain_dict:
            data_list = []
            for k in range(len(self.domain_dict[domain])):
                randind = np.random.randint(0, len(self.word_list))
                data_list.append(self.word_list[randind])

            self.domain_dict[domain] = data_list

    def average_similarity_word_vs_domain(self, word_one, given_list, start, end, step):
        wordsim = []
        for year in range(start, end, step):
            word_list = self.word2vec_pkl[str(Gender_Bias_Wiki.BEGIN_YEAR + year)]
            word_dic = dict({(x, i) for (i, x) in enumerate(word_list)})

            word_vec = self.word2vec_npy[str(Gender_Bias_Wiki.BEGIN_YEAR + year)]

            similarity = []
            for word_two in given_list:  # ["lesbian"]
                try:
                    vec_one = np.array(word_vec[word_dic[word_one]])
                    vec_two = np.array(word_vec[word_dic[word_two]])
                except:
                    continue

                sim = cosine_similarity([vec_one], [vec_two])
                similarity.append(sim[0][0])

            wordsim.append(np.average(similarity))
        return wordsim

    def gender_vs_domains(self, word):
        domain_similarity = {}

        for domain in self.domain_dict:
            word_list = self.domain_dict[domain]
            avg_sim = self.average_similarity_word_vs_domain(word, word_list, self.start, self.end, self.step)
            domain_similarity[domain] = avg_sim

        return domain_similarity

    def return_gender_stats(self, gender_list):

        gender_association = {}

        for word in gender_list:
            domain_similarity = self.gender_vs_domains(word)
            gender_association[word] = domain_similarity

        return gender_association

    def create_data_store_stats(self):

        self.data_store = {}
        self.data_store[self.female_domain[0]] = self.return_gender_stats(self.female_domain[1:])
        self.data_store[self.male_domain[0]] = self.return_gender_stats(self.male_domain[1:])
