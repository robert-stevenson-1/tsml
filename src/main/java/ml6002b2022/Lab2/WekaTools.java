package ml6002b2022.Lab2;

import org.nd4j.linalg.util.ArrayUtil;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public abstract class WekaTools {
    public static double accuracy(Classifier c, Instances test) throws Exception {
        double accuracy = 0.0;
        for (Instance i :
                test) {
            if(i.classValue() == c.classifyInstance(i)){
                accuracy++;
            }
        }
        accuracy /= test.numInstances();
        return accuracy;
    }

    public static Instances loadClassificationData(String fullPath){
        Instances instances = null;
        try{
            FileReader reader = new FileReader(fullPath);
            instances = new Instances(reader);
            instances.setClassIndex(instances.numAttributes()-1);
        }catch(Exception e){
            System.out.println("Exception caught: " + e);
        }
        return instances;
    }

    public static Instances[] splitData(Instances all, double proportion){
        Random r = new Random(System.nanoTime()); //seed the RNG with system nano time
        //round the value for the proportion of test instances
        int testAmount = (int)Math.round(all.numInstances()*proportion);
        Instances test = new Instances(all, 0);
        Instances train = new Instances(all, 0);

        while(testAmount > 0){
            int i = r.nextInt(all.numInstances());
            test.add(all.instance(i)); //add the instance to the test set
            all.remove(i); //remove that instance from the list
            //decrement value
            testAmount--;
        }
        train.addAll(all);
        return new Instances[]{train, test};
    }

    public static double[] classDistribution(Instances data){
        int classNum = data.numClasses();
        double[] classQuantity = new double[classNum];

        for (Instance instance:
             data) {
            int i = (int) instance.classValue();
            classQuantity[i]++;
            System.out.println(i);
        }

        for (int i = 0; i < classNum; i++) {
            classQuantity[i] = classQuantity[i]/data.numInstances();
        }

        return classQuantity;
    }

    public static int[][] confusionMatrix(int[] predicted, int[] actual, int numClasses){
        int[][] matrix = new int[numClasses][numClasses];
        for (int i = 0; i < actual.length; i++) {
            matrix[actual[i]][predicted[i]]++;
        }
        return matrix;
    }

    public static int[] classifyInstances(Classifier c, Instances test) throws Exception {
        ArrayList<Integer> predicted = new ArrayList<>();

        for (Instance inst :
                test) {
            predicted.add((int) c.classifyInstance(inst));
        }
        return ArrayUtil.toArray(predicted);
    }

    public static int[] getClassValues(Instances data){
        ArrayList<Integer> actual = new ArrayList<>();
        for (Instance inst :
                data) {
            actual.add((int)inst.classValue());
        }
        return ArrayUtil.toArray(actual);
    }

    private static void Task2_D() throws Exception {
        String dataPath = ".\\src\\main\\java\\ml6002b2022\\Lab2\\Arsenal_TEST.arff";
        String trainDataPath = ".\\src\\main\\java\\ml6002b2022\\Lab2\\Arsenal_TRAIN.arff";
        Instances data = loadClassificationData(dataPath);
        Instances trainData = loadClassificationData(trainDataPath);

        System.out.println("|=| kNN TRAINED MODEL:");
        IBk kNN = new IBk(); //kNN classifier but defaults to 1 neighbour
        kNN.buildClassifier(trainData);

        int[] pred = classifyInstances(kNN, data);
        int[] actual = getClassValues(data);
        System.out.println("Actuals: " + Arrays.toString(actual));
        System.out.println("pred:    " + Arrays.toString(pred));
        System.out.println("Confusion Matrix: " + Arrays.deepToString(confusionMatrix(pred, actual, data.numClasses())));
        System.out.println("Accuracy: " + accuracy(kNN, data));

        System.out.println("\n ITALY POWER DEMAND \n");

        dataPath = "src/main/java/experiments/data/tsc/ItalyPowerDemand/ItalyPowerDemand_TEST.arff";
        trainDataPath = "src/main/java/experiments/data/tsc/ItalyPowerDemand/ItalyPowerDemand_TRAIN.arff";
        data = loadClassificationData(dataPath);
        trainData = loadClassificationData(trainDataPath);

        System.out.println("|=| kNN TRAINED MODEL:");
        kNN = new IBk(); //kNN classifier but defaults to 1 neighbour
        kNN.buildClassifier(trainData);

        pred = classifyInstances(kNN, data);
        actual = getClassValues(data);
        System.out.println("Actuals: " + Arrays.toString(actual));
        System.out.println("pred:    " + Arrays.toString(pred));
        System.out.println("Confusion Matrix: " + Arrays.deepToString(confusionMatrix(pred, actual, data.numClasses())));
        System.out.println("Accuracy: " + accuracy(kNN, data));
    }

    //test helper functions
    private static void test() throws Exception {
        String dataPath = ".\\src\\main\\java\\ml6002b2022\\Lab2\\Arsenal_TEST.arff";
        String trainDataPath = ".\\src\\main\\java\\ml6002b2022\\Lab2\\Arsenal_TRAIN.arff";
        Instances data = loadClassificationData(dataPath);
        Instances trainData = loadClassificationData(trainDataPath);
        System.out.println(data);

        Instances[] split = new Instances[2];
        split = splitData(data, 0.5);
        System.out.println("Test: \n" + split[0]);
        System.out.println("Train: \n" + split[1]);

        double[] classDist = classDistribution(data);
        System.out.println("Class Count: " + Arrays.toString(classDist));

        int[] actual = new int[] {0,0,1,1,1,0,0,1,1,1};
        int[] predicted = new int[] {0,1,1,1,1,1,1,1,1,1};

        System.out.println("Actual: " + Arrays.toString(actual));
        System.out.println("Predicted: " + Arrays.toString(predicted));
        System.out.println("Confusion Matrix: " + Arrays.deepToString(confusionMatrix(predicted, actual, data.numClasses())));

        J48 classifier = new J48();
        classifier.buildClassifier(trainData);

        System.out.println("Classify Instances: " + Arrays.toString(classifyInstances(classifier, data)));

        System.out.println("Actual class values: " + Arrays.toString(getClassValues(data)));
    }

    public static void main(String[] args) throws Exception {
        Task2_D();
    }
}
