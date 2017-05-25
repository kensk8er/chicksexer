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

* Sometimes simple name lookup does not work. For instance, "Ryu" is likely to be a male name if it's followed by a Japanese surname, whereas it can be a Korean surname as well, then it's gender neutral.
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
'neutral'
>>> predict_gender('Ponya', return_proba=False)  # modify the name such that it sounds like a female name
'female'
>>> predict_gender('Ryu Ito', return_proba=True)  # Ryu here is a Japanese first name
{'female': 0.04139333963394165, 'male': 0.9586066603660583}
>>> predict_gender('Ryu Seo-yeon', return_proba=False)  # Ryu is a Korean surname, Seo-yeon is a popular first name for girls
{'female': 0.7503564655780792, 'male': 0.24964353442192078}
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
