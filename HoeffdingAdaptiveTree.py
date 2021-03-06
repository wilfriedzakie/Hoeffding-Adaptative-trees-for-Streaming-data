from skmultiflow.classification.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.classification.naive_bayes import NaiveBayes
from skmultiflow.classification.core.driftdetection.adwin import ADWIN
from skmultiflow.classification.core.utils.utils import do_naive_bayes_prediction
from abc import ABCMeta, abstractmethod
from skmultiflow.core.utils.utils import *
import math
import random
import numpy as np


class HoeffdingAdaptiveTree(HoeffdingTree):

    """ Hoeffding AdaptiveTree 

    A Hoeffding Adaptive tree is a decision tree-like algorithm which extends Hoeffding tree algorithm. 
    It's used for learning incrementally from data streams. 
    It grows tree as is done by the the Hoeffding Tree Algorithm and has also as mathematical guarantee the Hoeffding bound. 

    It builds alternate trees whenever a some changes in the distribution is noticed by ADWIN. 
    Unlike other adaptives window tree Algorithms, such as Hoeffding tree window, it does not need a fixed size of window before to raise an alarm.
    Statistics are store in nodes which decide themselves the number of sufficient examples they needs to split.  
    The alternate tree can promoted if the tree decrease in accuracy compared to the alternate one.

    See for details:
    https://link.springer.com/chapter/10.1007r%2F978-3-642-03915-7_22

    Implementation based on:
    MOA: Massive Online Analysis; Journal of Machine Learning Research 11: 1601-1604


    
    """

    class NewNode(metaclass= ABCMeta):
        """ Abstract Class whose methods are overridden by other subclasses methods """
        @abstractmethod
        def numberLeaves(self):
            pass

        @abstractmethod
        def getErrorEstimation(self):
            pass

        @abstractmethod
        def getErrorWidth(self):
            pass

        @abstractmethod
        def isNullError(self):
            pass

        @abstractmethod
        def killTreeChilds(self, ht):
            pass

        @abstractmethod
        def learnFromInstance(self, X,y, weight, ht,parent, parentBranch):
            pass

        @abstractmethod
        def filterInstanceToLeaves(self, X, myparent, parentBranch,updateSplitterCounts,foundNodes):
            pass


    def __init__(self,*args, **kwarg):
        """HoeffdingTree class constructor."""
        super(HoeffdingAdaptiveTree,self).__init__(*args, **kwarg)
        self.treeRoot = None
        self._pruned_alternate_trees = 0
        self._switch_alternate_trees = 0
        self.alternateTree=0
        

    class AdaSplitNode(HoeffdingTree.SplitNode,NewNode):
        """ 
        Class for a node that split the data in a hoeffding Adaptive tree
        Parameters
        ----------
        :param split_test: used Instantiate Hoeffding Tree SplitNode class 
		:type split_test:InstanceConditionalTest.
        :param class_observations: Class observations
        :type class_observations: dict (class_value, weight) or None.    
        :param size: Number of splits
        :type  size: int. 

         """


        def __init__(self, split_test, class_observations, size):
            """SplitNode class constructor"""
            HoeffdingTree.SplitNode.__init__(self, split_test,class_observations,size)
            self.numLeaves = 0
            self.estimationErrorWeight = ADWIN()
            self.ErrorChange = False
            self.alternateTree = None
            self.randomSeed = 1
            self.classifierRandom = random.seed(self.randomSeed)

        def numberLeaves(self):
            """Calculate number of node’s children leaves
            Returns
            -------
            num_of_leaves:int
                Number of node's leaves


            """
            num_of_leaves = 0
            for child in self._children:
                if child is not None:
                    num_of_leaves += child.number_leaves()

            return num_of_leaves

        
        def calc_byte_size_including_subtree(self):
            """Calculate the size of the node including its subtree
            
            Returns
            -------
            byteSize:int
                Size of the node and its subtree in bytes

            """

            byteSize = self.__sizeof__()

            if self.alternateTree is not None:
                byteSize += self.alternateTree.calc_byte_size_including_subtree()
            if self.estimationErrorWeight is not None:
                byteSize += self.estimationErrorWeight.get_length_estimation()

            for child in self._children:
                    if child is not None:
                        byteSize += child.calc_byte_size_including_subtree()
            return byteSize

        
        def number_leaves(self):
            """Calculate number of node’s leaves
            Returns
            -------
            num_of_leaves:int
                Number of node's leaves

            """
            numleaves = 0
            for child in self._children:
                if child is not None:
                    numleaves += child.number_leaves()
            return numleaves

        def getErrorEstimation(self):
            """
			Get the skmultiflow ADWIN class distribution error estimation
            Returns
            -------
            num_of_leaves:int
                Number of node's leaves

            """
            
            return self.estimationErrorWeight._estimation

           
        def getErrorWidth(self):
            """ 
            Get the skmultiflow ADWIN class distribution error width
            Returns
            -------
            w:float
                error width

			"""
            w = 0.0
            if self.isNullError() is False:
                w = self.estimationErrorWeight._width
            return w

        
        def isNullError(self):
            """ 
            Verify  skmultiflow ADWIN class distribution contains error 
            Returns
            -------
            self.estimationErrorWeight:boolean                         
            """
            return self.estimationErrorWeight is None



        
        def learnFromInstance(self, X, y, weight, ht, parent, parent_branch):
            """
			Update the node with the provided instance and create alternate tree if significant change is detected in class distribution  .

            Parameters
            ----------
            X: numpy.ndarray of length equal to the number of features.
                Instance attributes for updating the node.
            y: int
                Instance class.
            weight: float
                Instance weight.
            ht: Hoeffding Adaptive Tree
                Hoeffding Adaptive Tree to update.
            parent: HoeffdingTree.SplitNode or None
                The node's parent.
            parent_branch: int
                The parent node's branch

            """

            class_prediction = 0
            #k = np.random.poisson(1.0,self.classifierRandom)
            if self.filter_instance_to_leaf(X, parent, parent_branch).node is not None:
                 res=self.filter_instance_to_leaf(X, y, parent, parent_branch).node.get_class_votes(X, ht)

                 #Get the majority vote
                 max=0
                 maxIdx=0
                 for k ,v in res.items():
                     if v>max:
                         maxIdx=k
                         max = 0
                 class_prediction =maxIdx

            bl_correct = (y == class_prediction)

            if self.estimationErrorWeight is None:
                self.estimationErrorWeight = ADWIN()

            old_error = self.get_error_estimation()

            self.estimationErrorWeight.add_element(0.0 if bl_correct == True else 1.0)
            self.error_change =self.estimationErrorWeight.detected_change()
            

            if self.error_change == True and old_error > self.get_error_estimation():
                self.error_change = False

            if self.error_change ==True:
                self.alternateTree = ht.new_learning_node()
                ht.alternateTrees += 1

            
            elif self.alternateTree is not None and self.alternateTree.is_null_error() == False:
                if self.get_error_width() > 300 and self.alternateTree.get_error_width() > 300:
                    old_error_rate = self.get_error_estimation()
                    alt_error_rate = self.alternateTree.get_error_estimation()
                    fDelta = 0.05
                    fn = 1.0 / (self.alternateTree.get_error_width()) + 1.0 / (self.get_error_width())

                    bound = 1.0 / math.sqrt(2.0 * old_error_rate * (1.0 - old_error_rate) * math.log(2.0 / fDelta)*fn)

                    if bound < old_error_rate - alt_error_rate:
                        ht._active_leaf_node_cnt -= self.numberLeaves()
                        ht._active_leaf_node_cnt += self.alternateTree.numberLeaves()
                        self.killTreeChilds(ht)

                        if parent is not None:
                            parent.setChild(parent_branch, self.alternateTree)
                        else:
                            ht.treeRoot = ht.treeRoot.alternateTree
                        ht._switchAlternateTrees += 1

                    elif bound < alt_error_rate - old_error_rate:
                        if isinstance(self.alternateTree, HoeffdingTree.ActiveLearningNode):
                            self.alternateTree = None
                            self._active_leaf_node_cnt -= 1

                        elif isinstance(self.alternateTree, HoeffdingTree.InactiveLearningNode):#tricky
                            self.alternateTree = None
                            self._inactive_leaf_node_cnt -= 1

                        else:
                            self.alternateTree.killTreeChilds(ht)
                        ht._prunedalternateTree += 1

            if self.alternateTree is not None:
                self.alternateTree.learnFromInstance(X, y, weight, ht, parent, parent_branch)

            child_branch = self.instance_child_index(X)
            child = self.get_child(child_branch)

            if child is not None:
                child.learnFromInstance(X, y, weight, ht, parent, parent_branch) #tricky

        
        def killTreeChilds(self, ht):
            """
			Remove from tree node's children

            Parameters
            ----------
            ht: Hoeffding Adaptive Tree
                Hoeffding Adaptive Tree to remove childre.
        
            """
            for child in self._children:
                if child is not None:

                    if isinstance(child, HoeffdingAdaptiveTree.AdaSplitNode):
                        child.killTreeChilds(ht)

                    if isinstance(child, HoeffdingAdaptiveTree.ActiveLearningNode):
                        child = None
                        ht._active_leaf_node_cnt -= 1


                    elif isinstance(child, HoeffdingTree.InactiveLearningNode):
                        child = None
                        ht._inactive_leaf_node_cnt -= 1
                        

        def filterInstanceToLeaves(self, X, myparent, parentBranch,updateSplitterCounts,foundNodes=None):
            """Travers down the tree to locate the corresponding leaf for an instance.

             Parameters
             ----------
             X: Data instances.
                    myparent:
                            Parent node.
                        parent_branch: Int
                            Parent branch index
                        updateSplitterCounts: Boolean
                        FoundNode: List

            :param X:
            :param myparent: HoeffdingTree.Node
            :param parentBranch:
            :param updateSplitterCounts:
            :param foundNodes:
            """

            if foundNodes is None:
               foundNodes = []
            child_index = self.instance_child_index(X)

            if child_index >= 0:
                child = self.get_child(child_index)

                if child is not None:
                    child.filterInstanceToLeaves(X, myparent,parentBranch, updateSplitterCounts,foundNodes)
                else:
                    foundNodes.append(HoeffdingTree.FoundNode(None, self, child_index))

            if self.alteranteTree is not None:
                 self.alternateTree.filterInstanceToLeaves(X, self,  -999, updateSplitterCounts,foundNodes)

            
    def _new_learning_node(self, initial_class_observations=None):
        """Create a new instnce of the AdaLearningNode class

                  Parameters
                  ----------
                  initial_class_observations: ADWIN


        """
        return self.AdaLearningNode(initial_class_observations)


    def new_split_node(self, split_test, class_observations, size):

        """Create a new instance of the AdaSplitNode class

                Parameters
                ----------
        			initial_class_observations: ADWIN


        """

        return self.AdaSplitNode(split_test, class_observations, size)


    class AdaLearningNode(HoeffdingTree.LearningNodeNBAdaptive, NewNode):

            """ class for Learning Nodes in a Hoeffding Adaptive Tree.

                Parameters
                ----------
                initial_class_observations: dict (class_value, weight) or None
                    Initial class observations

            """
            
            def __init__(self, initialClassObservations):
                HoeffdingTree.LearningNodeNBAdaptive.__init__(self, initialClassObservations)
                self.estimationErrorWeight= ADWIN()
                self.ErrorChange = False
                self.randomSeed = 1
                self.classifierRandom = random.seed(self.randomSeed)

           
            def calcByteSize(self):
                """Calculate the size of the node including its subtree.
            
                Returns
                -------
                byteSize:int
                    Size of the node and its subtree in bytes.

                """
                byteSize = self.__sizeof__()
                if self.estimationErrorWeight is not None:
                    byteSize += self.estimationErrorWeight.get_length_estimation()
                return byteSize

         
            def numberLeaves(self):
                """ Return 1"""
                return 1

            
            def getErrorEstimation(self):
                """ Get the skmultiflow ADWIN class distribution error estimation
                Returns
                -------
                num_of_leaves:int
                    Number of node's leaves

                """
                if self.estimationErrorWeight is not None:
                    return self.estimationErrorWeight._estimation
                else:
                    return 0

           
            def getErrorWidth(self):
                """ Get the skmultiflow ADWIN class distribution error width
                    Returns
                    -------
                    w:float
                        error width

                """
                return self.estimationErrorWeight._width

           
            def isNullError(self):
                """ 
                Verify  skmultiflow ADWIN class distribution contains error 
                Returns
                -------
                self.estimationErrorWeight:boolean                         
                """
                return self.estimationErrorWeight is None

          
            def killTreeChilds(self, ht):
                """Remove from tree node's children"""
                pass

           
            def learnFromInstance(self, X, y, weight, ht, parent, parentBranch):
                """Update the node with the provided instance and create alternate tree if significant change is detected in class distribution  .

                Parameters
                ----------
                X: numpy.ndarray of length equal to the number of features.
                    Instance attributes for updating the node.
                y: int
                    Instance class.
                weight: float
                    Instance weight.
                ht: Hoeffding Adaptive Tree
                    Hoeffding Adaptive Tree to update.
                parent: HoeffdingTree.SplitNode or None
                    The node's parent.
                parent_branch: int
                    The parent node's branch

                """
                ClassPrediction = 0

                k = np.random.poisson(1.0, self.classifierRandom)

                if (k > 0):
                    weight = weight * k

                vote = self.get_class_votes(X, ht)

                # Get the majority vote
                max = 0
                maxIdx = 0
                for k, v in vote.items():
                     if v > max:
                         maxIdx = k
                         max=v
                ClassPrediction = maxIdx


                blCorrect = (y == ClassPrediction)

                if (self.estimationErrorWeight is None):
                    self.estimationErrorWeight = ADWIN()

                oldError = self.getErrorEstimation()

                self.estimationErrorWeight.add_element(0.0 if blCorrect == True else 1.0)
                self.ErrorChange = self.estimationErrorWeight.detected_change()


                if self.ErrorChange == True and oldError > self.getErrorEstimation():
                    self.ErrorChange = False

                super().learn_from_instance(X, y, weight, ht)

                weight_seen = self.get_weight_seen()

                if weight_seen - self.get_weight_seen_at_last_split_evaluation() >= ht.grace_period:
                    ht._attempt_to_split(self, parent, parentBranch)
                    self.set_weight_seen_at_last_split_evaluation(weight_seen)

           
            
            def normalize(self,sum, dist):
                """Normalizes all numeric values in the given dataset (apart from the class attribute, if set). 
                The resulting values are by default in [0,1]

                Parameters
                ----------
                dist: Dictionnary
                    Instance values
                """
                if sum == 0 and math.isnan(sum):
                    for key, value in dist.items(): 
                        value = value / sum


            
            def get_class_votes(self, X, ht):
             
                """Get class votes for a single instance.

                Parameters
                ----------
                X: numpy.ndarray of length equal to the number of features.
                Instance attributes.

                Returns
                -------
                dict (class_value, weight)
                """
                
                dist = {}
                
                if (self._mc_correct_weight > self._nb_correct_weight):
                        dist = self.get_observed_class_distribution()
                else:
                        dist = do_naive_bayes_prediction(X, self._observed_class_distribution, self._attribute_observers)

                distSum = sum(dist.values())
                distSum=0
                for key in dist:
                    distSum+=dist[key]

                ds = distSum * self.getErrorEstimation() * self.getErrorEstimation()
                if ds > 0.0:
                    self.normalize(ds, dist)
                return dist

           
            def filterInstanceToLeaves(self, X, parent, parent_branch,
                                        updateSplitterCounts,foundNodes=None):
                                        
                """
                Travers down the tree to locate the corresponding leaf for an instance.

                Parameters
                ----------
                X: Data instances.
                parent: HoeffdingTree.Node
                    Parent node.
                parent_branch: Int
                    Parent branch index
                updateSplitterCounts: Boolean
                FoundNode: List
                """
                
                if foundNodes is None:
                    foundNodes=[]
                foundNodes.append(HoeffdingTree.FoundNode(self, parent, parent_branch))

            def new_split_node(split_test, class_observations):
                return HoeffdingAdaptiveTree.AdaSplitNode(split_test, class_observations)

    #|---------------------------------------------------------------------------------------------------------------#
    
    def _new_learning_node(self, initial_class_observations=None):
        """Learn New Nodes"""
        return self.AdaLearningNode(initial_class_observations)

    
    def filterInstanceToLeaves(self, X, split_parent, parent_branch, update_splitter_counts):
        """Travers down the tree to locate the corresponding leaf for an instance.

            Parameters
            ----------
            X: Data instances.
            parent: HoeffdingTree.Node
                Parent node.
            parent_branch: Int
                Parent branch index
            updateSplitterCounts: Boolean
            nodes: List
            
            Returns
            ----------
            nodes: List
            """
        nodes = []
        self.treeRoot.filterInstanceToLeaves(X, split_parent, parent_branch, update_splitter_counts, nodes)
        return nodes



    
    


    def partial_fit(self, X, y, classes=None, weight=None):
        
        """ partial_fit
        Incrementally trains the model. The weight of the instances re computed and passed to _partial_fit

        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            Data instances.

        y: Array-like
            Contains the classification targets for all samples in X.

        classes: Not used.

        weight: Float or Array-like
            Instance weight. If not provided, uniform weights are assumed.

        Returns
        -------
        self

        """

        if y is not None:
            if weight is None:
                weight = np.array([1.0])
            row_cnt, _ = get_dimensions(X)
            wrow_cnt, _ = get_dimensions(weight)
            if row_cnt != wrow_cnt:
                weight = [weight[0]] * row_cnt
            for i in range(row_cnt):
                if weight[i] != 0.0:
                    self._partial_fit(X[i], y[i], weight[i])


    def _partial_fit(self, X, y, weight):
        """ _partial_fit
                Incrementally trains the model. Train samples (instances) are compossed of X attributes, their
                corresponding targets y and their weight.

                Parameters
                ----------
                X: Numpy.ndarray of shape (n_samples, n_features)
                    Data instances.

                y: Array-like
                    Contains the classification targets for all samples in X.

                classes: Not used.

                weight: Float or Array-like
                    Instance weight. If not provided, uniform weights are assumed.

                Returns
                -------
                self

        """

        if self.treeRoot is None:
            self.treeRoot = self._new_learning_node()
            self._active_leaf_node_cnt = 1
        self.treeRoot.learnFromInstance(X, y, weight, self, None, -1)




    def predict(self, X):
        """Predicts the label of the X instance(s)
         Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Samples for which we want to predict the labels.
        Returns
        -------
        list
            Predicted labels for all instances in X.

        """

        
        r, _ = get_dimensions(X)
        predictions = []
        for i in range(r):
            votes = self.get_votes_for_instance(X[i])
            if votes == {}:
                # Tree is empty, all classes equal, default to zero
                predictions.append(0)
            else:
                predictions.append(max(votes, key=votes.get))
        return predictions
    
    def get_votes_for_instance(self, X):
        if self.treeRoot is not None:
            found_nodes = self.filterInstanceToLeaves(X, None, -1, False)
            result = {}
            predictionPaths = 0

            for found_node in found_nodes:
                if found_node.parent != -999:
                    leaf_node = found_node.node
                    if leaf_node is None:
                        leaf_node = found_node.parent
                    dist = leaf_node.get_class_votes(X, self)
                    result.update(dist)
            return result
        else:
            return {}
