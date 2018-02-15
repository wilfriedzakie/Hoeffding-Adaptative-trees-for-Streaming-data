.. Hoeffding Adaptive Tree documentation master file, created by
   sphinx-quickstart on Thu Feb 15 02:04:47 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Hoeffding Adaptive Tree roject's documentation!
===================================================

A Hoeffding Adaptive tree is a decision tree-like algorithm which extends Hoeffding tree algorithm. 
It's used for learning incrementally from data streams. 
It grows tree as is done by the the Hoeffding Tree Algorithm and has also as mathematical guarantee the Hoeffding bound. 
It builds alternate trees whenever a some changes in the distribution is noticed by ADWIN. Unlike other adaptives window tree Algorithms, such as Hoeffding tree window, it does not need a fixed size of window before to raise an alarm.

Statistics are stored in nodes which decide themselves the number of sufficient examples they needs to split.  
The alternate tree can promoted if the tree decrease in accuracy compared to the alternate one.

This project was an IOT Stream Data Mining course's project of the Master 2 Data and Knowledge (Telecom Paristech & Université Paris-Saclay) 
under the supervision of Professor Albert Bifet and Jacob Montiel.



.. toctree::
   :maxdepth: 3
   :caption: Contents:

   Installation
   Documentation
   

Indices and tables
==================
* :ref:`Installation`
* :ref:`Documentation`
