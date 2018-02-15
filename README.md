# Hoeffding Adaptative Tree for streaming data

#### Developed by Wilfried ZAKIE & Wafa Djerad 

   A Hoeffding Adaptive tree is a decision tree-like algorithm which extends Hoeffding tree algorithm. 
It's used for learning incrementally from data streams. 
It grows tree as is done by the the Hoeffding Tree Algorithm and has also as mathematical guarantee the Hoeffding bound. 
It builds alternate trees whenever a some changes in the distribution is noticed by ADWIN. Unlike other adaptives window tree Algorithms, such as Hoeffding tree window, it does not need a fixed size of window before to raise an alarm.

Statistics are store in nodes which decide themselves the number of sufficient examples they needs to split.  
The alternate tree can promoted if the tree decrease in accuracy compared to the alternate one.

See for details: [https://link.springer.com/chapter/10.1007r%2F978-3-642-03915-7_22](https://link.springer.com/chapter/10.1007r%2F978-3-642-03915-7_22)

Documentation:
Installation:


Implementation based on: MOA: Massive Online Analysis; Journal of Machine Learning Research 11: 1601-1604
