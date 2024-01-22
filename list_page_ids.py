import getopt
import logging
import os
import pickle
import sys
import time
from gensim.corpora.wikicorpus import WikiCorpus

""" This code was created by Katja Schmahl. This code is licensed using a [Creative Commons License](http://creativecommons.org/licenses/by/4.0/). 
This means you can use, edit and distribute it as long as you give appropriate credit and 
indicate if changes were made."""

logger = logging.getLogger(__name__)
timestamp = time.strftime('%Y%m%d-%H%M')


def main(args):
    if not os.path.exists('logs/list-page-ids/'):
        os.makedirs('logs/list-page-ids')
    log_fname = f"logs/list-page-ids/{timestamp}.log"
    logging.basicConfig(filename=log_fname,
                        level=logging.INFO,
                        filemode='w',
                        format='%(asctime)s %(levelname)-8s %(message)s')
    global logger
    logger = logging.getLogger(__name__)

    try:
        opts, args = getopt.getopt(args, "hi:y:", ["help", "ifile=", "year="])
    except getopt.GetoptError:
        print('Use list_page_ids.py -y <year> -i <inputfile>')
        sys.exit(2)

    if '-h' in args:
        print('Use list_page_ids.py -y <year> -i <inputfile>')
        sys.exit(1)

    else:
        inp, year = None, None
        for opt, arg in opts:
            if opt in ('-i', '--ifile'):
                inp = arg
            elif opt in ('-y', '--year'):
                year = arg

        if year is None:
            print('Year should be given as a argument')
            sys.exit(2)

        list_page_ids(inp, year)


def list_page_ids(inp, year):
    if os.path.exists(f'data/ids/ids-{year}'):
        return
    corpus = WikiCorpus(inp, dictionary={}, filter_namespaces=False)
    corpus.metadata = True
    ids = []
    counter = 0
    logger.info(f"Saving pageid - {year}")
    for (text, (pageid, title)) in corpus.get_texts():
        if counter % 10000 == 0:
            logger.info(f"Saved {counter} pageids")
        ids.append(int(pageid))
        counter += 1

    logger.info(f"DONE: counted {counter} pageids")
    if not os.path.exists('data/ids'):
        os.mkdir('data/ids')
    file = open(f'data/ids/ids-{year}', 'wb')
    pickle.dump(ids, file)
    logger.info(f"DONE: Saved {counter} pageids")


if __name__ == "__main__":
    main(sys.argv[1:])
