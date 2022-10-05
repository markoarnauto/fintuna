Usage
=====

.. _installation:

Installation
------------

To use Fintuna, first install it using pip:

.. code-block:: console

   (.venv) $ pip install fintuna

Creating recipes
----------------

To explore the performance of a model run ``fintuna.explore()`` function:

.. autofunction:: fintuna.explore

The ``model`` parameter should be a :py:class:`fintuna.model.BaseModel`.

For example:

>>> import fintuna
>>> fintuna.explore()

