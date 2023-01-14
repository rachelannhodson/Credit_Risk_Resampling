# Credit Risk Resampling -- README.md

This is a Jupyter Lab notebook file. This file utilizes Python machine learning to predict future true/false (0/1) type outcomes. Specifically, with datasets in which the data is rather imbalanced. For instance, is this loan healthy (0) or high-risk (1)? A data analyst can upload their `.csv` file and run a logistics regression prediction to determine how well their model is predicting outcomes. They may then run a resampled (oversampled) predition and cross-reference the two models. Side-by-side, the data analyst will have a much better idea as to how well their model is prediction outcomes. Especially in situations where one of the options has far less data values than the other.

---

## Technologies

Please be sure you have Jupyter Lab installed:

* [JupyterLab](https://jupyter.org/)

This application was written in Python 3.9.12. This application is dependent on the following libraries:

* [numpy](https://numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [pathlib](https://docs.python.org/3/library/pathlib.html)
* [sklearn metrics balanced accuracy score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)
* [sklearn metrics confusion matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
* [imblearn metrics classification report imbalanced](https://imbalanced-learn.org/dev/references/generated/imblearn.metrics.classification_report_imbalanced.html)

---

## Installation Guide

If you have [Anaconda](https://www.anaconda.com/products/distribution) downloaded, then pandas and matplotlib will be part of your package. You can check that they're ready to use by typing the following in your CLI terminal:

```python
conda list numpy
conda list pandas
conda list pathlib
```

You can download the following local machine using this code:
```python
pip install -U scikit-learn
conda install -c conda-forge imbalanced-learn
```

And check that they've been installed with this code:
```python
conda list scikit-learn
conda list imbalanced-learn
```

---

## Usage

Open your CLI terminal and type
```python
jupyter lab
```
then JupyterLab will automatically open in your browswer. Use the left side menu bar to search for the `credit_risk_resampling` file. Open this file. Then you can use the formaulas in the `.ipynb` file to analyze your `.csv` file(s) and analyze your data.

---

## Contributors

[Rachel Ann Hodson](https://www.linkedin.com/in/rachelannhodson/)
rachelannhodson@gmail.com

---

## License

MIT