.. pgmtwin documentation master file, created by
   sphinx-quickstart on Thu Jul 24 15:53:22 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. |libname| replace:: pgmtwin

Welcome to |libname|'s documentation!
=====================================

.. image:: ../../readme/pgmtwin_logo.png
   :alt: pgmtwin
   :align: center
   :width: 300px

|libname| is a Python library for building, simulating, and analyzing digital twins.

We provide wrappers and algorithms to help a user set up the a loop of

- simulation of a physical asset, whose state might not be directly available to other components
- update of a digital asset's state, according to the results of an inverse problem on the physical asset's observable features
- evaluation of a policy to select the action to apply to the physical asset

.. image:: ../../readme/digital_twin.svg
   :alt: Digital Twin Overview
   :align: center
   :width: 550px

The following image represents the workflow for the development of a new digital twin system using |libname|.
Please refer to the tutorial notebooks and examples for implementation references.

.. image:: ../../readme/pgmtwin_schema.svg
   :alt: |libname| Workflow
   :align: center
   :width: 600px

We assume that the user provides a simulation for the evolution of the physical asset's state, 
triggered by an action and able to generate observations.

A first step is thus the definition of the (possibly specialized) 
:class:`BaseAction <pgmtwin.core.action.BaseAction>`\ s and a 
:class:`DiscreteDomain <pgmtwin.core.domain.DiscreteDomain>` for the digital asset's state 
which should mirror the physical asset's. \
At the moment, |libname| only supports discrete domains for the digital asset.

The final implementation task is the inverse problem to transform a set of observations 
of the physical asset into a probability distribution of the digital asset's state.

At this point, the :class:`BasePhysicalAsset <pgmtwin.core.physical_asset.BasePhysicalAsset>` 
and :class:`BaseDigitalAsset <pgmtwin.core.digital_asset.BaseDigitalAsset>` classes can be extended 
to wrap the simulation and inverse problem procedures, and 
a new :class:`BaseDigitalTwinEnv <pgmtwin.core.env.BaseDigitalTwinEnv>` can be created to manage 
the simulation.

The :class:`BaseDigitalTwinEnv <pgmtwin.core.env.BaseDigitalTwinEnv>` extends the Env class 
from `gymnasium <https://gymnasium.farama.org/>`_, enabling compatibility with policy training
frameworks such as `stable-baselines3 <https://stable-baselines3.readthedocs.io/>`_.

|libname| also provides some domain-specific implementations in the toolkits module. \ 
For example, the Structural Health Monitoring module :mod:`shm <pgmtwin.toolkits.shm>` provides a 
custom :class:`SingleDamageDomain <pgmtwin.toolkits.shm.domain.SingleDamageDomain>`
, :class:`MaintenanceAction <pgmtwin.toolkits.shm.action.MaintenanceAction>`
and :class:`DigitalTwinEnv <pgmtwin.toolkits.shm.env.DigitalTwinEnv>`
to recreate the setup in the 
paper `A digital twin framework for civil engineering structures <https://doi.org/10.1016/j.cma.2023.116584>`_

**Get started by exploring the API documentation:**

:doc:`API Reference <pgmtwin>`

----

.. toctree::
   :maxdepth: 2

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
