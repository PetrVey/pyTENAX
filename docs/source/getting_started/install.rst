Installation
============`

For the moment the package is not available on pypi, so you need to install it from the source code.

To do so, clone the repository:

.. code-block:: bash

    git clone https://github.com/PetrVey/pyTENAX.git
  
With Conda, run the following command in the root folder of the repository.

.. code-block:: bash

    # create pytenax environment
    conda env create -f env.yml
    # activate pytenax environment
    conda activate pytenax_env
    # install pytenax in editable mode
    python -m pip install -e .


