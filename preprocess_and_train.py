import getopt
import os
import pickle
import sys
import logging
import time
import random
import re

from evaluate_quality import evaluate_quality
from gensim.models import Word2Vec
import numpy as np
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.word2vec import LineSentence
from gensim.corpora.wikicorpus import WikiCorpus
from list_page_ids import list_page_ids

""" This code was created by Katja Schmahl. This code is licensed using a [Creative Commons License](http://creativecommons.org/licenses/by/4.0/). 
This means you can use, edit and distribute it as long as you give appropriate credit and 
indicate if changes were made."""

logger = logging.getLogger(__name__)
timestamp = time.strftime('%Y%m%d-%H%M')


def main(args):
    if not os.path.exists('logs/preprocess-train'):
        os.makedirs('logs/preprocess-train')
    log_fname = f'logs/preprocess-train/{timestamp}.log'
    logging.basicConfig(filename=log_fname,
                        level=logging.INFO,
                        filemode='w',
                        format='%(asctime)s %(levelname)-8s %(message)s')

    global logger
    logger = logging.getLogger(__name__)
    try:
        opts, args = getopt.getopt(args, "hi:y:lefn:srp",
                                   ["help", "ifile=", "year=", "load", "evaluate", "filter", "iterations=", "save-iterations", "random", "preprocess"])
    except getopt.GetoptError:
        print('Use preprocess_and_train.py -y <year> [-i <inputfile> -l -e -f]')
        print('Option -i / --ifile to specify location of dump, e.g. -i data/raw/2008.xml.bz2')
        print('Option -l / --load if you want to load an already trained model to evaluate')
        print('Option -e / --evaluate if you want to do a quality evaluation')
        print('Option -f / --filter if you want to use the filtered on new articles, '
              'make sure you want have the ids listed before')
        sys.exit(2)

    if '-h' in args:
        print('Use preprocess_and_train.py -y <year> [-i <inputfile> -l -e -f]')
        print('Option -i / --ifile to specify location of dump, e.g. -i data/raw/2008.xml.bz2')
        print('Option -l / --load if you want to load an already trained model to evaluate')
        print('Option -e / --evaluate if you want to do a quality evaluation')
        print('Option -f / --filter if you want to use the filtered on new articles, '
              'make sure you want have the ids listed before')
        sys.exit(1)

    else:
        inp, years, load, do_eval, filtered, iterations, save_iterations, use_random_seed, skip_preprocess = \
            None, [], False, False, False, 5, True, False, False
        for opt, arg in opts:
            print(opts)
            if opt in ('-i', '--ifile'):
                inp = [i.strip() for i in arg.split(',')]
            elif opt in ('-y', '--year'):
                years = [int(i.strip()) for i in arg.split(',')]
                print(years)
            elif opt in ('-l', '--load'):
                load = True
            elif opt in ('-e', '--evaluate'):
                do_eval = True
            elif opt in ('-f', '--filter'):
                filtered = True
            elif opt in ('-n', '--iterations'):
                iterations = int(arg)
            elif opt in ('-s', '--save-iterations'):
                save_iterations = True
            elif opt in ('-r', '--random'):
                use_random_seed = True
            elif opt in ('-p', '--preprocess'):
                print('-p')
                skip_preprocess = True
        if years is None:
            print('Year should be given as a argument')
            sys.exit(2)

        if use_random_seed:
            seed = random.randint(0, 2**32)
        else:
            seed = 0

        if inp is None:
            if skip_preprocess:
                inp = [None for year in years]
            else:
                inp = [f"data/raw/{year}.xml.bz2" for year in years]
        print(years)
        print(inp)
        print(list(zip(years, inp)))

        for (year, ix) in zip(years, inp):
            print(year)
            print(ix)

        for year, input_file in zip(years, inp):
            print(input_file)
            preprocess_and_train_data(inp=input_file, year=year, load=load, filtered=filtered,
                                      iterations=iterations, seed=seed, save_iterations=save_iterations)

        if do_eval:
            evaluate_quality(years=years, iterations=iter, filtered=filtered)


def preprocess_and_train_data(inp, year, load, filtered, iterations, seed, save_iterations):
    logger.info(f'Preprocessing and training with input={inp}, year={year}, load={load}, filtered={filtered}, seed={seed}')

    if filtered:
        txt_file = f'data/corpus/corpus_{year}_filtered.txt'
    else:
        txt_file = f'data/corpus/corpus_{year}.txt'
    
    if not os.path.exists('data/corpus'):
        os.makedirs('data/corpus')    
    if not os.path.exists(txt_file):
        open(txt_file, 'w').close()

    logger.info(f'Trying to create corpus with file: {inp}')

    # CREATE CORPUS FROM RAW DUMP
    if inp is not None:
        if not filtered:
            preprocess(inp, year)
        else:
            list_page_ids(inp, year)
            preprocess_filter(inp, year)

    # TRAIN MODEL OR LOAD EXISTING
    if load:
        directory_path = f'data/models/{"filtered" if filtered else "regular"}/iter{iterations}'
        model = Word2Vec.load(f'{directory_path}/{year}{"_filtered" if filtered else ""}_iter{iterations}')

    else:
        if save_iterations:
            model = Word2Vec(LineSentence(open(txt_file, 'r')), sample=0, workers=3, max_vocab_size=100000, epochs=iterations, vector_size=100, seed=seed, callbacks=[SaveIterationsCallback(year, filtered)])
        else:
            model = Word2Vec(LineSentence(open(txt_file, 'r')), sample=0, workers=3, max_vocab_size=100000,
                             epochs=iterations, vector_size=100, seed=seed)

            save_model_in_stereotype_format(model=model, filtered=filtered, iterations=iterations,
                                            year_string=f"{year}_{'filtered_' if filtered else ''}iter{iterations}")


class SaveIterationsCallback(CallbackAny2Vec):

    def __init__(self, year, filtered=False):
        self.year = year
        self.filtered = filtered
        self.iterations = 1

    def on_epoch_end(self, model):
        logger.info('EPOCH END')
        file_name = f'{self.year}{"_filtered" if self.filtered else ""}_iter{self.iterations}'
        save_model_in_stereotype_format(model, self.filtered, self.iterations, file_name)
        self.iterations = self.iterations + 1
        logger.info(f'Iteration {self.iterations} saved as {file_name}')

    @classmethod
    def on_batch_begin(cls, model):
        return

    @classmethod
    def on_batch_end(cls, model):
        return

    @classmethod
    def on_epoch_begin(cls, model):
        logger.info('EPOCH BEGIN')
        return

    @classmethod
    def on_train_begin(self, model):
        logger.info('TRAIN BEGIN')
        return

    @classmethod
    def on_train_end(self, model):
        logger.info('TRAIN END')
        return



filter_categories = []

def build_filter(categories):
    for cat in categories:
        filter_categories.append(re.compile(r'^\[\[Category:' + re.escape(cat) + r'\]\]$', flags=re.MULTILINE))


def filter_articles(elem, text, *args, **kwargs):
    if text is None:
        return False
    for cat in filter_categories:
        if cat.search(text):
            return True
    return False


def preprocess(inp, year):
    txt_file = f'data/corpus/corpus_{year}.txt'
    
    build_filter([ 'Science', 'Scientists', 'Biography', 'Woman' , 'Engineering', 'Man'])
    filter_categories.append(re.compile(r'^\[\[Category:.*(S|s)cientists.*\]\]$', flags=re.MULTILINE))
    filter_categories.append(re.compile(r'^\[\[Category:.*(I|i)nventors.*\]\]$', flags=re.MULTILINE))
    filter_categories.append(re.compile(r'^\[\[Category:.*(T|t)echnology.*\]\]$', flags=re.MULTILINE))
    filter_categories.append(re.compile(r'^\[\[Category:.*(E|e)ngineer.*\]\]$', flags=re.MULTILINE))
    filter_categories.append(re.compile(r'^\[\[Category:.*(R|r)esearch.*\]\]$', flags=re.MULTILINE))
    filter_categories.append(re.compile(r'^\[\[Category:.*(A|a)cademic.*\]\]$', flags=re.MULTILINE))
    
    corpus = WikiCorpus(inp, dictionary={}, filter_namespaces=False, filter_articles=filter_articles)
    save_corpus_as_txt(corpus, txt_file)


def preprocess_filter(inp, year):
    txt_file = f'data/corpus/corpus_{year}_filtered.txt'
    fc = FilterClass(year -1)
    corpus = WikiCorpus(inp, dictionary={}, filter_articles=fc.filter_articles_pageid(inp, year -1))
    save_corpus_as_txt(corpus, txt_file)


def save_model_in_stereotype_format(model, filtered, iterations, year_string='2024'):
    dictionary = dict()
    print("year_string", year_string)
    for index, word in enumerate(model.wv.index_to_key):
        dictionary[word] = index
    if filtered:
        directory = f'data/models/filtered/iter{iterations}'
    else:
        directory = f'data/models/regular/iter{iterations}'

    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = f'{directory}/{year_string}'
    pickle_file = open(f'{file_path}-vocab.pkl', 'wb')
    model.save(f'{file_path}')
    np.save(f'{file_path}.vectors.wv', model.wv.vectors)
    pickle.dump(dictionary, file=pickle_file)

def save_corpus_as_txt(corpus, fname):
    # corpus.metadata = True
    
    output = open(fname, 'w')
    count = 0
    for text in corpus.get_texts():
        output.write(' '.join(text) + '\n')
        output.flush()

        count += 1
        if (count % 10000 == 0):
            logger.info("Saved " + str(count) + " articles")

    logger.info(f"Total articles: {count}")


class FilterClass:
    def __init__(self, year):
        self.ids = set(np.load(f'data/ids/ids-{year}', allow_pickle=True))
        logger.info("Amount of ids is {l}".format(l=len(self.ids)))
        self.count_rejected = 0
        self.count_accepted = 0

    def filter_articles_pageid(self, elem, namespace, **kwargs):
        ns_mapping = {"ns": namespace}
        pageid_path = "./{%(ns)s}id" % ns_mapping
        if isinstance(elem, int) and elem in self.ids:
            self.count_rejected += 1
            if self.count_rejected % 10000 == 0:
                logger.info("Rejected {cr} articles, accepted {ca}".format(cr=self.count_rejected,
                                                                           ca=self.count_accepted))
            return False
        else:
            self.count_accepted += 1
            return True


if __name__ == "__main__":
    # calling main function
    main(sys.argv[1:])
