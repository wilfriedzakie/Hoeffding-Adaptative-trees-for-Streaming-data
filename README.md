# Hoeffding Adaptative Tree for streaming data

<<<<<<< HEAD
#### Developed by Wilfried ZAKIE, Wafa Djerad && Kien DANG Trung
=======
#### Developed by Wilfried ZAKIE & Wafa Djerad 
>>>>>>> a5939f9c0f24b1ef398cf8944214be50ac3b1f25

   A Hoeffding Adaptive tree is a decision tree-like algorithm which extends Hoeffding tree algorithm. 
It's used for learning incrementally from data streams. 
It grows tree as is done by the the Hoeffding Tree Algorithm and has also as mathematical guarantee the Hoeffding bound. 
It builds alternate trees whenever a some changes in the distribution is noticed by ADWIN. Unlike other adaptives window tree Algorithms, such as Hoeffding tree window, it does not need a fixed size of window before to raise an alarm.

Statistics are store in nodes which decide themselves the number of sufficient examples they needs to split.  
The alternate tree can promoted if the tree decrease in accuracy compared to the alternate one.

See for details: [https://link.springer.com/chapter/10.1007/978-3-642-03915-7_22](https://link.springer.com/chapter/10.1007/978-3-642-03915-7_22)

Documentation:


Installation: This algorithm needs [scikit Multiflow](https://scikit-multiflow.github.io/) a stream data framework inspired by MOA and Weka



![Scikit Logo](https://scikit-multiflow.github.io/scikit-multiflow/_images/skmultiflow-logo-wide.png)





Implementation based on: MOA: Massive Online Analysis; Journal of Machine Learning Research 11: 1601-1604
