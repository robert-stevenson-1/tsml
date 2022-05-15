package ml_6002b_coursework;

import utilities.ClassifierTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.*;

import experiments.data.*;

import java.io.IOException;
import java.util.Arrays;

/**
 * A basic decision tree classifier for use in machine learning coursework (6002B).
 */
public class CourseworkTree extends AbstractClassifier {

    /** Measure to use when selecting an attribute to split the data with. */
    private AttributeSplitMeasure attSplitMeasure = new IGAttributeSplitMeasure();

    /** Maxiumum depth for the tree. */
    private int maxDepth = Integer.MAX_VALUE;

    /** The root node of the tree. */
    private TreeNode root;

    /**
     * Sets the attribute split measure for the classifier.
     *
     * @param attSplitMeasure the split measure
     */
    public void setAttSplitMeasure(AttributeSplitMeasure attSplitMeasure) {
        this.attSplitMeasure = attSplitMeasure;
    }

    /**
     * Sets the max depth for the classifier.
     *
     * @param maxDepth the max depth
     */
    public void setMaxDepth(int maxDepth){
        this.maxDepth = maxDepth;
    }

    /**
     * Returns default capabilities of the classifier.
     *
     * @return the capabilities of this classifier
     */
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);

        //instances
        result.setMinimumNumberInstances(2);

        return result;
    }

    /**
     * Builds a decision tree classifier.
     *
     * @param data the training data
     */
    @Override
    public void buildClassifier(Instances data) throws Exception {
        if (data.classIndex() != data.numAttributes() - 1) {
            throw new Exception("Class attribute must be the last index.");
        }

        root = new TreeNode();
        root.buildTree(data, 0);
    }

    /**
     * Classifies a given test instance using the decision tree.
     *
     * @param instance the instance to be classified
     * @return the classification
     */
    @Override
    public double classifyInstance(Instance instance) {
        double[] probs = distributionForInstance(instance);

        int maxClass = 0;
        for (int n = 1; n < probs.length; n++) {
            if (probs[n] > probs[maxClass]) {
                maxClass = n;
            }
        }

        return maxClass;
    }

    /**
     * Computes class distribution for instance using the decision tree.
     *
     * @param instance the instance for which distribution is to be computed
     * @return the class distribution for the given instance
     */
    @Override
    public double[] distributionForInstance(Instance instance) {
        return root.distributionForInstance(instance);
    }

    /**
     * Class representing a single node in the tree.
     */
    private class TreeNode {

        /** Attribute used for splitting, if null the node is a leaf. */
        Attribute bestSplit = null;

        /** Best Split value learned on construction for  numeric attribute data */
        private double bestSplitValue;

        /** Best gain from the splitting measure if the node is not a leaf. */
        double bestGain = 0;

        /** Depth of the node in the tree. */
        int depth;

        /** The node's children if it is not a leaf. */
        TreeNode[] children;

        /** The class distribution if the node is a leaf. */
        double[] leafDistribution;


        /**
         * Recursive function for building the tree.
         * Builds a single tree node, finding the best attribute to split on using a splitting measure.
         * Splits the best attribute into multiple child tree node's if they can be made, else creates a leaf node.
         *
         * @param data Instances to build the tree node with
         * @param depth the depth of the node in the tree
         */
        void buildTree(Instances data, int depth) throws Exception {
            this.depth = depth;

            // Loop through each attribute, finding the best one.
            for (int i = 0; i < data.numAttributes() - 1; i++) {
                double gain = attSplitMeasure.computeAttributeQuality(data, data.attribute(i));

                if (gain > bestGain) {
                    bestSplit = data.attribute(i);
                    bestGain = gain;
                    bestSplitValue = attSplitMeasure.getSplitVal(); // learn the best split value
                }
            }

            // If we found an attribute to split on, create child nodes.
            if (bestSplit != null) {
                Instances[] split;
                if (bestSplit.isNumeric()) {
                    split = attSplitMeasure.splitDataOnNumeric(data, bestSplit, 0);

                    // Added based on my interpretation from the guidance given in:
                    //   https://learn.uea.ac.uk/webapps/discussionboard/do/message?action=list_messages&course_id=_135859_1&nav=discussion_board&conf_id=_117994_1&forum_id=_14532618_1&message_id=_4585997_1#
                    //   from the Blackboard Discussion board question asked in 01/05/2022, and answered on 03/05/2022
                    bestSplitValue = attSplitMeasure.getSplitVal(); // learn the best split value

                }else{
                    split = attSplitMeasure.splitData(data, bestSplit);
                }
                children = new TreeNode[split.length];

                // Create a child for each value in the selected attribute, and determine whether it is a leaf or not.
                for (int i = 0; i < children.length; i++){
                    children[i] = new TreeNode();

                    boolean leaf = split[i].numDistinctValues(data.classIndex()) == 1 || depth + 1 == maxDepth;

                    if (split[i].isEmpty()) {
                        children[i].buildLeaf(data, depth + 1);
                    } else if (leaf) {
                        children[i].buildLeaf(split[i], depth + 1);
                    } else {
                        children[i].buildTree(split[i], depth + 1);
                    }
                }
            // Else turn this node into a leaf node.
            } else {
                leafDistribution = classDistribution(data);
            }
        }

        /**
         * Builds a leaf node for the tree, setting the depth and recording the class distribution of the remaining
         * instances.
         *
         * @param data remaining Instances to build the leafs class distribution
         * @param depth the depth of the node in the tree
         */
        void buildLeaf(Instances data, int depth) {
            this.depth = depth;
            leafDistribution = classDistribution(data);
        }

        /**
         * Recursive function traversing node's of the tree until a leaf is found. Returns the leafs class distribution.
         *
         * @return the class distribution of the first leaf node
         */
        double[] distributionForInstance(Instance inst) {
            // If the node is a leaf return the distribution, else select the next node based on the best attributes
            // value.
            if (bestSplit == null) {
                return leafDistribution;
            } else {
                if (inst.value(bestSplit) < bestSplitValue){
                    return children[0].distributionForInstance(inst);
                }else {
                    return children[1].distributionForInstance(inst);
                }
            }
        }

        /**
         * Returns the normalised version of the input array with values summing to 1.
         *
         * @return the class distribution as an array
         */
        double[] classDistribution(Instances data) {
            double[] distribution = new double[data.numClasses()];
            for (Instance inst : data) {
                distribution[(int) inst.classValue()]++;
            }

            double sum = 0;
            for (double d : distribution){
                sum += d;
            }

            if (sum != 0){
                for (int i = 0; i < distribution.length; i++) {
                    distribution[i] = distribution[i] / sum;
                }
            }

            return distribution;
        }

        /**
         * Summarises the tree node into a String.
         *
         * @return the summarised node as a String
         */
        @Override
        public String toString() {
            String str;
            if (bestSplit == null){
                str = "Leaf," + Arrays.toString(leafDistribution) + "," + depth;
            } else {
                str = bestSplit.name() + "," + bestGain + "," + depth;
            }
            return str;
        }
    }

    //Adjust the tree's split criterion by changing the split measures via the 'splitMeasureOpt' parameter
    public void setOptions(SPLIT_OPTIONS splitMeasureOpt){
        switch (splitMeasureOpt){
            case INFO_GAIN:
                setAttSplitMeasure(new IGAttributeSplitMeasure());
                break;
            case INFO_GAIN_RATIO:
                setAttSplitMeasure(new IGAttributeSplitMeasure(false));
                break;
            case CHI_SQUARED:
                setAttSplitMeasure(new ChiSquaredAttributeSplitMeasure());
                break;
            case GINI:
                setAttSplitMeasure(new GiniAttributeSplitMeasure());
                break;
        }
    }

    // Enum used for setting up the attribute slitting measure option
    public enum SPLIT_OPTIONS{
        INFO_GAIN,
        INFO_GAIN_RATIO,
        CHI_SQUARED,
        GINI
    }

    /**
     * Main method.
     *
     * @param args the options for the classifier main
     */
    public static void main(String[] args) throws Exception {
        // load data for test
        Instances optDigitsData = DatasetLoading.loadData("src/main/java/ml_6002b_coursework/test_data/optdigits.arff");

        CourseworkTree CWtreeIG = new CourseworkTree();
        CourseworkTree CWtreeIGRatio = new CourseworkTree();
        CourseworkTree CWtreeChiSquare = new CourseworkTree();
        CourseworkTree CWtreeGini = new CourseworkTree();

        System.out.println(
                "===============================================\n" +
                "=================optDigit Data=================\n" +
                "===============================================\n");
        CWtreeIG.setOptions(SPLIT_OPTIONS.INFO_GAIN);
        CWtreeIGRatio.setOptions(SPLIT_OPTIONS.INFO_GAIN_RATIO);
        CWtreeChiSquare.setOptions(SPLIT_OPTIONS.CHI_SQUARED);
        CWtreeGini.setOptions(SPLIT_OPTIONS.GINI);

        //split the data randomly in 1/2 into a training and test set via the WekaTools class split method
        Instances[] splitData = WekaTools.splitData(optDigitsData, 0.5); // index 1: Train, index 0: Test

        //build and train the classifiers
        CWtreeIG.buildClassifier(splitData[0]);
        CWtreeIGRatio.buildClassifier(splitData[0]);
        CWtreeChiSquare.buildClassifier(splitData[0]);
        CWtreeGini.buildClassifier(splitData[0]);

        // test the accuracy of the classifiers
        System.out.println("DT using measure <measureInformationGain> on optdigits problem has test accuracy = "
                + ClassifierTools.accuracy(splitData[1], CWtreeIG));
        System.out.println("DT using measure <measureInformationGainRatio> on optdigits problem has test accuracy = "
                + ClassifierTools.accuracy(splitData[1], CWtreeIGRatio));
        System.out.println("DT using measure <measureGini> on optdigits problem has test accuracy = "
                + ClassifierTools.accuracy(splitData[1], CWtreeGini));
        System.out.println("DT using measure <measureChiSquared> on optdigits problem has test accuracy = "
                + ClassifierTools.accuracy(splitData[1], CWtreeChiSquare));

        System.out.println(
                "===============================================\n" +
                "================Chinatown Data=================\n" +
                "===============================================\n");
        Instances chinatownData = DatasetLoading.loadData("src/main/java/ml_6002b_coursework/test_data/Chinatown.arff");


        CWtreeIG = new CourseworkTree();
        CWtreeIGRatio = new CourseworkTree();
        CWtreeChiSquare = new CourseworkTree();
        CWtreeGini = new CourseworkTree();

        CWtreeIG.setOptions(SPLIT_OPTIONS.INFO_GAIN);
        CWtreeIGRatio.setOptions(SPLIT_OPTIONS.INFO_GAIN_RATIO);
        CWtreeChiSquare.setOptions(SPLIT_OPTIONS.CHI_SQUARED);
        CWtreeGini.setOptions(SPLIT_OPTIONS.GINI);

        //split the data randomly in 1/2 into a training and test set via the WekaTools class split method
        Instances[] chinaSplitData = WekaTools.splitData(chinatownData, 0.5); // index 1: Train, index 0: Test

        //build and train the classifiers
        CWtreeIG.buildClassifier(chinaSplitData[0]);
        CWtreeIGRatio.buildClassifier(chinaSplitData[0]);
        CWtreeChiSquare.buildClassifier(chinaSplitData[0]);
        CWtreeGini.buildClassifier(chinaSplitData[0]);
        // test the accuracy of the classifiers
        System.out.println("DT using measure <measureInformationGain> on Chinatown problem has test accuracy = "
                + ClassifierTools.accuracy(chinaSplitData[1], CWtreeIG));
        System.out.println("DT using measure <measureInformationGainRatio> on Chinatown problem has test accuracy = "
                + ClassifierTools.accuracy(chinaSplitData[1], CWtreeIGRatio));
        System.out.println("DT using measure <measureGini> on Chinatown problem has test accuracy = "
                + ClassifierTools.accuracy(chinaSplitData[1], CWtreeGini));
        System.out.println("DT using measure <measureChiSquared> on Chinatown problem has test accuracy = "
                + ClassifierTools.accuracy(chinaSplitData[1], CWtreeChiSquare));


    }
}