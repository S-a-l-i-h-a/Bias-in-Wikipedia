import getopt
import os
import sys

import matplotlib.pyplot as plt

""" This code was created by Katja Schmahl. This code is licensed using a [Creative Commons License](http://creativecommons.org/licenses/by/4.0/). 
This means you can use, edit and distribute it as long as you give appropriate credit and 
indicate if changes were made."""

color = ["black", "green", "blue", "magenta", "brown", "cyan"]
marker = ['o', 's', 'p', 'd', '>', '<']
occupations = ["All", "Manager", "Scientist", "Artist"]

# Data obtained from denelezh.org
overall = [16.808, 16.911, 17.008, 17.104, 17.377, 17.538, 17.700, 17.641, 17.823, 17.778, 17.882, 18.058, 18.199, 18.351]
scientist = [10.690, 10.852, 11.080, 12.203, 12.462, 12.876, 14.736, 14.996, 15.651, 16.025, 15.934, 15.931, 16.267, 16.522]
manager = [9.552, 9.674, 9.761, 10.487, 10.287, 10.570, 11.121, 9.247, 10.449, 9.553, 11.581, 11.719, 11.656, 12.063]
artist = [28.508, 28.453, 28.715, 29.662, 28.914, 28.890, 29.057, 29.423, 29.467, 29.439, 28.917, 29.021, 29.137, 29.225]
lines = [overall, scientist, manager, artist]


def main(args):
    try:
        opts, args = getopt.getopt(args, "h", ["help"])
    except getopt.GetoptError:
        print('Argument error')
        print('Use python3 biographies_plot.py, without options>')
        sys.exit(2)

    if ('-h', '--help') in args:
        print('Use python3 biographies_plot.py, without options>')
        print('The results can be found in results/biographies.png:')
        sys.exit(1)

    else:
        biographies_plot()


def biographies_plot():
    labels = []
    for y in range(2017, 2021):
        for m in range(1, 13, 3):
            if y == 2020 and m > 5:
                break
            labels.append(f'{y}/{m}')

    plt.figure(figsize=(8, 6))
    for i in range(0, 4):
        plt.plot(lines[i], color=color[i], marker=marker[i], label=occupations[i])
    plt.title("Female biographies for different occupations", fontsize=20)
    # plt.xlabel('Date')
    plt.xticks([i for i in range(0, len(labels))], labels, rotation=45, fontsize=16)
    plt.ylabel('Percentage of biographies that are female', fontsize=16)
    plt.yticks([i for i in range(0, 36, 5)], [f"{i}%" for i in range(0, 36, 5)], fontsize=16)
    plt.grid()
    plt.legend(fontsize=16, ncol=2)
    if not os.path.exists('results'):
        os.mkdir('results')
    plt.savefig('results/biographies.png', bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
