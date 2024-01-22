# Bias in Wikipedia (updated)

This is an updated version of the following repository: https://gitlab.com/kschmahl/wikipedia-gender-bias-over-time/-/tree/master

The codebase and dependencies have been updated to `Python 3.10`, and a few issues in the code have been fixed.
## How to run
- (Optional, but recommended) Make a new Python *virtual environment*:
    - `python -m venv env`
    - `source env/bin/activate` (activate the environment in your terminal)
- Install dependencies:
    - `pip install -r requirements.txt`
- Create a folder called `raw` inside the `data` folder.
- Put all of your **yearly** Wikipedia dumps (`.xml.bz2`) into the `data/raw` folder
    - The naming format is `{year}.xml.bz2`, e.g. `2024.xml.bz2`
    - If you just have the XML, you can "zip" it by running `bzip2 filename.xml`
- Open the `compute_bias_all_years.py` file and navigate to line `104`. There, change the lists `years`, `filter_years`, `boxplot_years` and `filter_boxplot_years` to include the years you want to process.
    - Make sure that you have previously added `xml.bz2` files for all of the years.
- To pre-process a year dataset, run:
    - `python preprocess_and_train.py -y {year}`, replacing the `{year}` with the year number.
    - E.g. `python preprocess_and_train.py -y 2024`
    - **Repeat** this for all of your years.
- To run the bias evaluation for all years, run:
    - `python compute_bias_all_years.py -s`
- You can observe the progress of all scripts from the `logs` folder.
    - E.g. logs for the `preprocess-train` script will be in the `logs/preprocess-train` folder.
- After execution, you can observe the results in the `results` folder.

---
> WIP - Saliha Mustafić