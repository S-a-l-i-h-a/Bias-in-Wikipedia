Undergraduate Thesis: Unveiling Bias in Wikipedia through Machine Learning
==========================================================================

Overview
--------

Wikipedia, hailed for its ubiquity and accessibility, has been scrutinized for harboring biases. This undergraduate thesis embarks on a journey to investigate biases within Wikipedia's narrative, particularly focusing on gender, race, and ethnicity. The project employs advanced machine learning techniques to uncover and address these biases. By collecting a curated dataset from Wikipedia articles pertaining to specific contexts and content types, the study analyzes the content using Natural Language Processing (NLP) techniques. The examination extends to sentiment analysis to discern the tone of the articles, unveiling nuanced insights. The application of simple machine learning models, such as decision trees and logistic regression, is then leveraged to train models using the dataset alongside demographic data. This research seeks to contribute valuable insights to the development of tools and techniques that mitigate biases in online platforms.

Data Source and Model
---------------------

This thesis builds upon the groundwork laid by [K. Schmahl's repository](https://gitlab.com/kschmahl/wikipedia-gender-bias-over-time) which explores gender bias in Wikipedia over time. The codebase has undergone some of the enhancements, now compatible with Python 3.11, and notable bug fixes have been implemented.

Dataset
---------------------
Created out of the xml dumps form [Wikimedia dump page](https://dumps.wikimedia.org/enwiki/20240101/) and on [Wikipedia torrent page](https://meta.wikimedia.org/wiki/Data_dump_torrents#English_Wikipedia) for older versions, as the Wikimedia page with their mirror pages consists only of last 5 recent dumps which are updated at least every month. Each file is pretty heavy, 2023 year having 21GB of data or around 5 milion articles which are filtered to around 270K which we used for research 
Later in preprocesss_and_train.py it is filtered up to certain Categories using RegEx (regular expressions) and Wikicorpus library to fit our research plan of investugating gender bias of woman and man biographies in science and engineering. 

Current results including the filtered years [2010,2017,2023,2024] you may find the folder ['results'](https://github.com/S-a-l-i-h-a/Bias-in-Wikipedia/tree/main/results)


Running the Program
-------------------

### 1\. Setting up a Virtual Environment

It is recommended to create a new Python virtual environment to isolate dependencies. Execute the following commands:


`python -m venv env`
`source env/bin/activate`

### 2\. Installing Dependencies

Install the required dependencies using:


`pip install -r requirements.txt`

### 3\. Preparing Data

Create a 'raw' folder inside the 'data' directory and place yearly Wikipedia dumps (.xml.bz2) into it. The files should follow the naming convention: {year}.xml.bz2 (e.g., 2024.xml.bz2). If only XML files are available, they can be compressed using bzip2.

### 4\. Configuring Script

Open the `compute_bias_all_years.py` file and navigate to line 104. Adjust the lists `years`, `filter_years`, `boxplot_years`, and `filter_boxplot_years` to include the desired years for processing. Ensure that corresponding XML.bz2 files are present.

### 5\. Pre-processing and Training

To pre-process data for a specific year, run:


`python preprocess_and_train.py -y {year}`

Replace `{year}` with the desired year, e.g., `python preprocess_and_train.py -y 2024`. Repeat this step for all years.

### 6\. Running Bias Evaluation

To execute bias evaluation for all years, run:

`python compute_bias_all_years.py -s`

Observe the progress in the 'logs' folder, with logs specific to the preprocess-train script in 'logs/preprocess-train'. Post-execution, explore results in the 'results' folder.

Your exploration into uncovering bias in Wikipedia has now commenced. May your findings contribute to the ongoing pursuit of unbiased and inclusive online platforms.

## Connect with me  
<div align="center">
<a href="https://github.com/S-a-l-i-h-a" target="_blank">
<img src=https://img.shields.io/badge/github-%2324292e.svg?&style=for-the-badge&logo=github&logoColor=white alt=github style="margin-bottom: 5px;" />
<a href="https://linkedin.com/in/Saliha Mustafić" target="_blank">
<img src=https://img.shields.io/badge/linkedin-%231E77B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white alt=linkedin style="margin-bottom: 5px;" />
</a> 
</div>  
  

<br/>  

~~~
Created based on:
(https://gitlab.com/kschmahl/wikipedia-gender-bias-over-time)
JK.G. Schmahl ,T.J. Viering ,S. Makrodimitris ,A. Naseri Jahfar, D.M.J.Tax and M. Loog (2020)
Is Wikipedia succeeding in reducing gender bias? Assessing changes in gender bias in Wikipedia using word embeddings.
~~~

