package ml6002b2022.Lab2;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.FileReader;
import java.util.Arrays;

public abstract class Train {

    public static Instances loadData(String path){
        Instances instances = null;
        try{
            FileReader reader = new FileReader(path);
            instances = new Instances(reader);
        }catch(Exception e){
            System.out.println("Exception caught: " + e);
        }
        return instances;
    }

    public static void task5(Instances test, Instances train){
        System.out.println("Num of Instances: " + test.numInstances());
        System.out.println("Num of Attributes: " + test.numAttributes());

        //Win count
        int wins = 0;
        //Loop through the data and count the number of 'wins'
        for (Instance i : test)
        {
            //get current instance, 'i', value of the 4th attribute (index 3)
            if(i.value(i.attribute(3)) == 2){
                wins++;
            }
        }
        System.out.println("Num of Wins: " + wins);

        System.out.println("5th instance as a Double[]: " + Arrays.toString(test.instance(4).toDoubleArray()));

        //Print Training data
        System.out.println("\nTraining Data: \n" + train.toString());

        //remove saka from the data
        System.out.println("Removing Saka...");

        Remove remove = new Remove(); //create a remove filter
        remove.setAttributeIndices("3"); //select the index of the attribute that i want to remove (SAKA: 3)
        //tell it i want to remove the index specified (true would keep the index's specified)
        remove.setInvertSelection(false);
        try {
            //remove from test data
            remove.setInputFormat(test); //set the input format
            test = Filter.useFilter(test, remove);

            //remove from training data
            remove.setInputFormat(train);
            train = Filter.useFilter(train, remove);

            System.out.println("Removal Complete");

            System.out.println("\n\"New\" Test Data: \n" + test.toString());
            System.out.println("\n\"New\" Training Data: \n" + train.toString());
        } catch (Exception e) {
            e.printStackTrace();
        }

        //reload the training and test data
//        test.delete();
//        test = loadData(dataLocTest);
//        train.delete();
//        train = loadData(dataLocTrain);

        System.out.println("\nReloaded Test Data: \n" + test.toString());
        System.out.println("\nReloaded Training Data: \n" + train.toString());
    }

    public static void main(String[] args) throws Exception {
        String dataLocTest = "Arsenal_TEST.arff";
        String dataLocTrain = "Arsenal_TRAIN.arff";
        Instances test = loadData(dataLocTest);
        Instances train = loadData(dataLocTrain);

        //set the classification value (from the attribute in the data)
        test.setClassIndex(test.numAttributes()-1);
        train.setClassIndex(train.numAttributes()-1);

        //define the classified we will use
        NaiveBayes bayes = new NaiveBayes(); //Naive Bayes
        IBk kNN = new IBk(); //kNN classifier but defaults ti 1 neighbour

        System.out.println("|=| BAYES TRAINED MODEL:");
        bayes.buildClassifier(train);
        int numCorrect = 0;

        //run the trained BAYES model in the test data
        for (Instance inst : test) {
            double actual = inst.value(inst.attribute(3));
            double pred = bayes.classifyInstance(inst); //get the predicted values
            System.out.println("Actual: " + actual + " ||| Predicted: " + pred);
            if (pred == actual){ //Predicted it correctly
                numCorrect++;
            }
        }
        System.out.println("    Number of Correct predictions (BAYES): " + numCorrect);

        System.out.println("|=| kNN TRAINED MODEL:");
        numCorrect = 0;
        kNN.buildClassifier(train);
        //run the trained kNN model in the test data
        for (Instance inst : test) {
            double actual = inst.value(inst.attribute(3));
            double pred = kNN.classifyInstance(inst); //get the predicted values
            System.out.println("Actual: " + actual + " ||| Predicted: " + pred);
            System.out.println("Distibution of Inst: " + Arrays.toString(kNN.distributionForInstance(inst)));
            if (pred == actual){ //Predicted it correctly
                numCorrect++;
            }
        }
        System.out.println("    Number of Correct predictions (kNN): " + numCorrect);

        System.out.println("Distribution comparisons: \n");
        for (Instance inst : test){
            System.out.println("    " + inst);
            System.out.println("        (BAYES) Distribution of Inst: " + Arrays.toString(bayes.distributionForInstance(inst)));
            System.out.println("          (kNN) Distribution of Inst: " + Arrays.toString(kNN.distributionForInstance(inst)));
        }
    }
}
