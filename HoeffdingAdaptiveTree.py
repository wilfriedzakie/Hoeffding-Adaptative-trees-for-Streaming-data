from skmultiflow.classification.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.classification.core.driftdetection.adwin import ADWIN
from abc import ABCMeta,abstractmethod

    class NewNode(metaclass= ABCMeta):

        @abstractmethod
        def show(self): raise NotImplementedError




    class AdaSplitNode(HoeffdingTree.SplitNode):

        def __init__(self):
            self.estimationErrorWeight: ADWIN
            self.ErrorChange=False
            self.alternateTree:HoeffdingTree.Node

        def getErrorWidth(self):
            w = 0.0
            if (self.isNullError() == False):
                w = self.estimationErrorWeight.getWidth();
            return w;


        def getErrorEstimation(self):
            return self.estimationErrorWeight.getEstimation()


        def isNullError(self):
            return self.estimationErrorWeight is None
        
            class AdaSplitNode(HoeffdingTree.SplitNode):

        def __init__(self):
            self.estimationErrorWeight: ADWIN
            self.ErrorChange=False
            self.alternateTree:HoeffdingTree.Node

        def getErrorWidth(self):
            w = 0.0
            if (self.isNullError() == False):
                w = self.estimationErrorWeight.getWidth();
            return w;


        def getErrorEstimation(self):
            return self.estimationErrorWeight.getEstimation()


        def isNullError(self):
            return self.estimationErrorWeight is None


        def numberLeaves():

            numLeaves = 0;
            for (Node child: self.children):
                if (child != null):
                        numLeaves += ((NewNode) child).numberLeaves()
            return numLeaves + 1;


        def getErrorEstimation(self):
            return self.estimationErrorWeight.getEstimation()

        def learnfromInstance(self,X,y, ht, parent:HoeffdingTree.SplitNode, parentBranch:int):
            ClassPrediction=0
            if(HoeffdingTree.SplitNode.filter_instance_to_leaf(X,parent,parentBranch).node is None):
                ClassPrediction=HoeffdingTree.Node(HoeffdingTree.SplitNode.filter_instance_to_leaf(X,parent,parentBranch).node).get_class_votes(X,ht)

            blCorrect = (y == ClassPrediction)

            if self.estimationErrorWeight is None :
                self.estimationErrorWeight = ADWIN()
            oldError = self.getErrorEstimation()
            self.ErrorChange = self.estimationErrorWeight.setInput(0.0 if blCorrect == True else  1.0)

            if self.ErrorChange== True and oldError>self.getErrorEstimation():
                self.ErrorChange=False

            if (self.ErrorChange == True):
                self.alternateTree = ht.newLearningNode()
                ht.alternateTrees +=1
            ##To check
            elif self.alternateTree is None and ((NewNode) self.alternateTree).isNullError() == False :
                if self.getErrorWidth() > 300 and ((NewNode) self.alternateTree).getErrorWidth() > 300:
                    oldErrorRate = self.getErrorEstimation()
                    altErrorRate = ((NewNode) self.alternateTree).getErrorEstimation()
                    fDelta = .05
    
