chicksexer - Python package for gender classification
=================================================================

![Chicksexer](images/chicksexer.jpg?raw=true "Title")

`chicksexer` is a Python package that performs **gender classification**. It receives a string of person name and returns the probability estimate of its gender as follows:

```python
>>> from chicksexer import predict_gender
>>> predict_gender('John Smith')
{'female': 0.0027230381965637207, 'male': 0.9972769618034363}
```

Several merits of using the classifier instead of simply looking up known male/female names are:

* Sometimes simple name lookup does not work. For instance, "Miki" is a Japanese female name, but also a Croatian male name.
* Can predict the gender of a name that does not exist in the list of male/female names.
* Can deal with a typo in a name relatively easily.

You can also get an estimate as a simple string as follows:

```python
>>> predict_gender('Oliver Butterfield', return_proba=False)
'male'
>>> predict_gender('Naila Ata', return_proba=False)
'female'
>>> predict_gender('Saldivar Anderson', return_proba=False)
'neutral'
>>> predict_gender('Ponyo', return_proba=False)  # name of a character from the film
'male'
>>> predict_gender('Ponya', return_proba=False)  # modify the name such that it sounds like a female name
'female'
>>> predict_gender('Miki Suzuki', return_proba=True)  # Suzuki here is a Japanese surname so Miki is a female name
{'female': 0.9997618066990981, 'male': 0.00023819330090191215}
>>> predict_gender('Miki Adamić', return_proba=True)  # Adamić is a Croatian surname so Miki is a male name
{'female': 0.16958969831466675, 'male': 0.8304103016853333}
>>> predict_gender('Jessica')
{'female': 0.999996105068476, 'male': 3.894931523973355e-06}
>>> predict_gender('Jesssica')  # typo in Jessica
{'female': 0.9999851534785194, 'male': 1.4846521480649244e-05}
```

If you want to predict the gender of multiple names, use `predict_genders` (plural) function instead:

```python
>>> from chicksexer import predict_genders
>>> predict_genders(['Ichiro Suzuki', 'Haruki Murakami'])
[{'female': 3.039836883544922e-05, 'male': 0.9999696016311646},
 {'female': 1.2040138244628906e-05, 'male': 0.9999879598617554}]
>>> predict_genders(['Ichiro Suzuki', 'Haruki Murakami'], return_proba=False)
['male', 'male']
```

Installation
------------
- This repository can run on Ubuntu 14.04 LTS & Mac OSX 10.x (not tested on other OSs)
- Tested only on Python 3.5

`chicksexer` depends on [NumPy and Scipy](https://www.scipy.org/install.html), Python packages for scientific computing. You might need to have them installed prior to installing `chicksexer`.

You can install `chicksexer` by:

```bash
pip install chicksexer
```

`chicksexer` also depends on `tensorflow` package. In default, it tries to install the CPU-only version of `tensorflow`. If you want to use GPU, you need to install `tensorflow` with GPU support by yourself. (C.f. [Installing Tensorflow](https://www.tensorflow.org/install/))

Model Architecture
------------------
The gender classifier is implemented using Character-level Multilayer LSTM. The architecture is roughly as follows:

1. Character Embedding Layer
2. 1st LSTM Layer
3. 2nd LSTM Layer
4. Pooling Layer
5. Fully Connected Layer

The fully connected layer outputs the probability of a name bing a male name. For the details, look at `_build_graph()` method in `chicksexer/_classifier.py`, which implements the computational graph of the architecture in `tensorflow`.

Training Data
-------------
Names with gender annotation are obtained from the sources as follows:

* [Dbpedia Person Data](http://downloads.dbpedia.org/2015-10/core-i18n/en/persondata_en.tql.bz2)
* [Popular baby names in the US](https://www.ssa.gov/oact/babynames/limits.html)
* [Names dataset curated by Milos Bejda](https://mbejda.github.io/)
