=================
PyTENAX
=================

| PyTENAX contains a set of methods to apply the Non-Asymptotic statistical model for eXtreme return levels (TENAX).
| Presented in manuscript:
| Marra, F., Koukoula, M., Canale, A., & Peleg, N. (2023). 
| Predicting extreme sub-hourly precipitation intensification based on temperature shifts. 
| Hydrology and Earth System Sciences Discussions, 2023, 1-23.
| https://doi.org/10.5194/hess-28-375-2024

Original TENAX model coded in Matlab:
TEmperature-dependent Non-Asymptotic statistical model for eXtreme return levels (TENAX)
https://doi.org/10.5281/zenodo.8332232


pyTENAX repository also includes:
| *SMEV class (Simplified Metastatistical Extreme Value)*
| For more information about SMEV, please see manuscripts:
| Francesco Marra, Davide Zoccatelli, Moshe Armon, Efrat Morin.
| A simplified MEV formulation to model extremes emerging from multiple nonstationary underlying processes.
| Advances in Water Resources, 127, 280-290, 2019
| https://doi.org/10.1016/j.advwatres.2019.04.002
  
| Francesco Marra, Marco Borga, Efrat Morin.
| A unified framework for extreme sub-daily precipitation frequency analyses based on ordinary events. 
| Geophys. Res. Lett., 47, 18, e2020GL090209. 2020.
| https://doi.org/10.1029/2020GL090209 

pythonized version of SMEV downloaded from:
https://github.com/luigicesarini/pysmev

| *A test for the hypothesis: block maxima are samples from a parent distribution with Weibull tail*
| The test is described in: 
|  Marra F, W Amponsah, SM Papalexiou, 2023. 
| Non-asymptotic Weibull tails explain the statistics of extreme daily precipitation. 
| Adv. Water Resour., 173, 104388, 
| https://doi.org/10.1016/j.advwatres.2023.104388
| Matlab source code:
| https://zenodo.org/records/7234708

Installation
------------
For the moment the package is not available on pypi, so you need to install it from the source code.
To do so, clone the repository and run the following command in the root folder of the repository.
  
With Conda 

.. code-block:: bash

    # create pytenax environment
    conda env create -f environment.yml
    # activate pytenax environment
    conda activate pytenax_env
    # install pytenax in editable mode
    python -m pip install -e .

Usage
-----
Background and Jupyter notebook examples (available also in Binder) are included in our readthedocs.

For a complete example of how to use the class, run the file `test_tenax.py` in the `example` folder with the following command:

.. code-block:: python

    python example/test_tenax.py

Development
-----------
To build a development environment run:

With Conda 

.. code-block:: bash

    conda env create -f environment.yml
    conda activate pytenax_env
    python -m pip install -e .


Please work on a feature branch and create a pull request to the source branch.
To ensure formatting consistency, please install the pre-commit hooks by running:

.. code-block:: bash

    pre-commit install

If necessary to merge manually do so without fast forward:

.. code-block:: bash

    git merge --no-ff myfeature

Contributions
-------------

## How to Submit an Issue

We welcome your feedback and contributions! If you encounter a bug, have a feature request, or have any other issue you'd like to bring to our attention, please follow the steps below:

1. **Check for Existing Issues**: Before you submit a new issue, please check if a similar issue already exists in our [issue tracker](https://github.com/PetrVey/pyTENAX/issues). If you find an existing issue that matches your concern, you can contribute to the discussion by adding your comments or reactions.

2. **Open a New Issue**: If you don't find an existing issue that matches your concern, you can open a new one by following these steps:
   - Go to the [Issues](https://github.com/PetrVey/pyTENAX/issues) section of the repository.
   - Click on the **New Issue** button.
   - Select the appropriate issue template, if available.
   - Fill in the title and description with as much detail as possible. Include steps to reproduce the issue, the expected behavior, and the actual behavior. Providing screenshots or code snippets can be very helpful.
   - Submit the issue.

3. **Follow Up**: After you submit the issue, we might need more information from you. Please stay tuned for our comments and respond promptly if we request additional details.

### Issue Submission Guidelines

- **Be Clear and Descriptive**: Help us understand the issue quickly and thoroughly.
- **Provide Context**: Describe the problem, including the version of the software, operating system, and any other relevant details.
- **Include Screenshots and Logs**: If applicable, add any screenshots, logs, or stack traces that can help diagnose the problem.
- **Use a Consistent and Descriptive Title**: This helps others quickly identify issues that might be similar to theirs.
- **Be Respectful and Considerate**: Keep in mind that we are all part of a community and we aim to create a positive and collaborative environment.

Thank you for helping us improve!

[Open an Issue](https://github.com/PetrVey/pyTENAX/issues/new)


Credits
-------
We wish to thank Riccardo Ciceri riccardo.ciceri@studenti.unipd.it for the first stage in developing pyTENAX 