package ml6002b2022.Lab5;

import experiments.data.DatasetLoading;
import ml6002b2022.Lab2.WekaTools;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IB1;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;

public class PlayerProblem {
    public static void main(String[] args) throws Exception {
        Instances data = DatasetLoading.loadData("C:\\Users\\Rls20\\" +
                "Documents\\GitHub\\tsml\\src\\main\\java\\ml6002b2022\\Lab5\\FootballPlayers.arff");
        System.out.println("No. Attributes: " + data.numAttributes());
        System.out.println("No. Instances: " + data.numInstances());
        System.out.println("No. Classes: " + data.numClasses());
        double[] classDistribution = WekaTools.classDistribution(data);
        System.out.println("Class Values: [Failure, Success]");
        System.out.println("Class Distribution: " + Arrays.toString(classDistribution));

        Instances[] splitData = InstanceTools.resampleInstances(data,0, 0.7);
        Instances train = splitData[0], test = splitData[1];

        // put classifiers in an arralist to be able to iterate through using each one more easily
        ArrayList<AbstractClassifier> classifiers = new ArrayList<>();
        classifiers.add(new OneNN());
        classifiers.add(new IB1());
        classifiers.add(new IBk());
        //classifiers.add(new LinearRegression());
        classifiers.add(new MultilayerPerceptron());
        classifiers.add(new SMO());

        //build all the classifier i'll be using on the training data
//        for (AbstractClassifier c :
//                classifiers) {
//            c.buildClassifier(train);
//        }

        //use classifiers on test data
        double[] accur = new double[classifiers.size()];
        for (int i = 0; i < classifiers.size(); i++) {
            //build the classifier on the training data
            classifiers.get(i).buildClassifier(train);
            //get the accuracy
            accur[i] = ClassifierTools.accuracy(test, classifiers.get(i));
            System.out.println("Classifer: " + classifiers.get(i).getClass().getSimpleName() + ", Accuracy: " + accur[i]);
        }

    }
}
