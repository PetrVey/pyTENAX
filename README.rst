=================
PyTENAX
=================

PyTENAX contains a set of methods to apply the Non-Asymptotic statistical model for eXtreme
return levels (TENAX).
Presented in manuscript:
Marra, F., Koukoula, M., Canale, A., & Peleg, N. (2023). Predicting extreme sub-hourly precipitation intensification based on temperature shifts. Hydrology and Earth System Sciences Discussions, 2023, 1-23.
https://doi.org/10.5194/hess-28-375-2024

Original model coded in Matlab:
TEmperature-dependent Non-Asymptotic statistical model for eXtreme return levels (TENAX)
https://zenodo.org/records/8345905


Installation
------------
Install using:

Python version required: <3.12

Module required can be found in either requirements.txt or env.yml

For the moment the package is not available on pypi, so you need to install it from the source code.
To do so, clone the repository and run the following command in the root folder of the repository:

.. code-block:: bash
    
   pip install .

Usage
-----
!! TO COMPLETE !!

The class contains the following methods:

!! TO COMPLETE !!

The following is an example of how to use the class:

.. code-block:: python

    #TODO



For a complete example of how to use the class, run the file `test_tenax.py` in the `src` folder with the following command:

.. code-block:: python

    python src/test_smev.py



Development
-----------
Please work on a feature branch and create a pull request to the development 
branch. If necessary to merge manually do so without fast forward:

.. code-block:: bash

    git merge --no-ff myfeature

To build a development environment run:

.. code-block:: bash

    python3 -m venv env 
    source env/bin/activate 
    pip install -e .
    pip install -r requirements.txt

With Conda 

.. code-block:: bash

    conda env create -f env.yml
    conda activate env
    pip install -e .


Rebuild is done by setup.py inside of new branch.
.. code-block::
    setup.py sdist bdist_wheel

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