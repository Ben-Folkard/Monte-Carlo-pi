Getting Started
===============

Installation
------------
.. code-block:: bash

   git clone https://github.com/Ben-Folkard/Monte-Carlo-pi.git
   cd Monte-Carlo-pi
   pip install -r requirements.txt

Running the Script
------------------
There's only 2 variables to mess around with:

- num_samples
- seed

You can either run it online via a GitHub action:

`GitHub Action Workflow <https://github.com/Ben-Folkard/Monte-Carlo-pi/actions/workflows/run_pi_calculation.yml>`_

Or you can download it and run it locally using the installation instructions above and then following the instructions below to run each of the files:

In serial:

.. code-block:: bash

   python3 calculating_pi_serial.py
   
   
In serial but passing the variables via the command line:


.. code-block:: bash

   python3 calculating_pi_serial_action_runnable.py --num_samples 1e5 --seed 12345
   

In parallel with njit:

.. code-block:: bash

   python3 calculating_pi_njit.py
   
   
In parallel with mpi4py:

.. code-block:: bash

   mpiexec -n 4 python3 calculating_pi_mpi.py