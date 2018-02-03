public class HoeffdingAdaptiveTree extends HoeffdingTree {
private static final long serialVersionUID = 1L;

    public String getPurposeString() {
         return "Hoeffding Adaptive Tree for evolving data streams that uses ADWIN to replace branches for new ones.";
  }

 public interface NewNode {

public static class AdaSplitNode extends SplitNode implements NewNode {

public int calcByteSizeIncludingSubtree() {

public AdaSplitNode(InstanceConditionalTest splitTest,

public int numberLeaves() {

public double getErrorEstimation() {

public double getErrorWidth() {

public boolean isNullError() {

public void learnFromInstance(Instance inst, HoeffdingAdaptiveTree ht, SplitNode parent, int parentBranch) {

public void killTreeChilds(HoeffdingAdaptiveTree ht) {

public void filterInstanceToLeaves(Instance inst, SplitNode myparent,

 public static class AdaLearningNode extends LearningNodeNBAdaptive implements NewNode {

private static final long serialVersionUID = 1L;

protected ADWIN estimationErrorWeight;

public boolean ErrorChange = false;

protected int randomSeed = 1;

protected Random classifierRandom;

public int calcByteSize() {

public AdaLearningNode(double[] initialClassObservations) {

public int numberLeaves() {

public double getErrorEstimation() {

ublic double getErrorWidth() {

public boolean isNullError() {


public void killTreeChilds(HoeffdingAdaptiveTree ht) {


public void learnFromInstance(Instance inst, HoeffdingAdaptiveTree ht, SplitNode parent, int parentBranch) {


public double[] getClassVotes(Instance inst, HoeffdingTree ht) {

public void filterInstanceToLeaves(Instance inst, SplitNode splitparent, int parentBranch, List<FoundNode> foundNodes, boolean updateSplitterCounts) {

protected int activeLeafNodeCount;
 
protected int inactiveLeafNodeCount;
 
protected int alternateTrees;

protected int prunedAlternateTrees;

protected int switchedAlternateTrees;

protected LearningNode newLearningNode(double[] initialClassObservations) {

protected SplitNode newSplitNode(InstanceConditionalTest splitTest,

public void trainOnInstanceImpl(Instance inst) {

public FoundNode[] filterInstanceToLeaves(Instance inst,

public double[] getVotesForInstance(Instance inst) {
