package ml6002b2022.Lab3;

import core.contracts.Dataset;
import experiments.data.DatasetLoading;
import scala.Array;
import utilities.ClassifierTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.LogitBoost;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;

import java.io.*;
import java.util.ArrayList;
import java.util.Date;

public class LabThreeExamples {

    static String[] problems={
            "bank",
            "blood",
            "breast-cancer-wisc-diag",
            "breast-tissue",
            "cardiotocography-10clases",
            "conn-bench-sonar-mines-rocks",
            "conn-bench-vowel-deterding",
            "ecoli",
            "glass",
            "hill-valley",
            "image-segmentation",
            "ionosphere",
            "iris",
            "libras",
            "optical",
            "ozone",
            "page-blocks",
            "parkinsons",
            "planning",
            "post-operative",
            "ringnorm",
            "seeds",
            "spambase",
            "statlog-landsat",
            "statlog-vehicle",
            "steel-plates",
            "synthetic-control",
            "twonorm",
            "vertebral-column-3clases",
            "wall-following",
            "waveform-noise",
            "wine-quality-white",
            "yeast"};

    public static void BaggingVsRandFrstExperiment() throws Exception {

        ArrayList<double[]> avr = new ArrayList<>();

        Instances train, test;

        for (String s :
                problems) {
            System.out.println("Problem: " + s);

            String trainPath = "C:\\Users\\Rls20\\Documents\\GitHub\\tsml\\src\\main\\" +
                    "java\\ml6002b2022\\Lab3\\UCIContinuous\\"+ s + "\\"+ s + "_TRAIN.arff";
            String testPath = "C:\\Users\\Rls20\\Documents\\GitHub\\tsml\\src\\main\\" +
                    "java\\ml6002b2022\\Lab3\\UCIContinuous\\"+ s + "\\"+ s + "_TEST.arff";

            //load the data
            train = DatasetLoading.loadData(trainPath);
            test = DatasetLoading.loadData(testPath);

            //create classifiers
            Bagging bagging = new Bagging();
            RandomForest randomForest = new RandomForest();

            //build the classifier
            bagging.buildClassifier(train);
            randomForest.buildClassifier(train);

            //get the accuracy of the classifiers
            double[] acc = new double[2];
            acc[0] = ClassifierTools.accuracy(test, bagging);
            acc[1] = ClassifierTools.accuracy(test, randomForest);

            System.out.print("  Bagging Acc: " + acc[0]);
            System.out.println("    Rand Forest Acc: " + acc[1]);
            avr.add(acc);
        }

        System.out.println("Writing to File....");

        try {

            String csvPath = "C:\\Users\\Rls20\\Documents\\GitHub" +
                    "\\tsml\\src\\main\\" +
                    "java\\ml6002b2022\\Lab3\\" +
                    "Averages_" + java.util.UUID.randomUUID() +".csv";
            File csvFile = new File(csvPath);

            if (!csvFile.createNewFile()){
                throw new Exception("ERROR: CSV File already exists with that name");
            }

            FileWriter fileWriter = new FileWriter(csvFile);
            BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);
            PrintWriter writer = new PrintWriter(bufferedWriter);

            StringBuilder sb = new StringBuilder();

            sb.append("baggingAcc");
            sb.append(",");
            sb.append("randFrstAcc");
            sb.append("\n");

            writer.write(sb.toString());

            for (double[] d:
                 avr) {
                sb.setLength(0); //clear the string builder
                sb.append(d[0]);
                sb.append(",");
                sb.append(d[1]);
                sb.append("\n");
                writer.write(sb.toString());
            }
            sb.setLength(0);
            sb.append("\n");
            sb.append("bagging Avr Acc");
            sb.append(",");
            sb.append("rand Forest Avr Acc");
            sb.append("\n");
            sb.append("=AVERAGE(A2:A34)");
            sb.append(",");
            sb.append("=AVERAGE(B2:B34)");
            sb.append("\n");
            writer.write(sb.toString());

            writer.close();
            System.out.println("Done Writing File");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

    }

    public static void BaggingVsRandSecondExperiment() throws Exception {

        ArrayList<double[]> accuracies = new ArrayList<>();

        Instances train, test;

        for (String s :
                problems) {
            System.out.println("Problem: " + s);

            String trainPath = "C:\\Users\\Rls20\\Documents\\GitHub\\tsml\\src\\main\\" +
                    "java\\ml6002b2022\\Lab3\\UCIContinuous\\"+ s + "\\"+ s + "_TRAIN.arff";
            String testPath = "C:\\Users\\Rls20\\Documents\\GitHub\\tsml\\src\\main\\" +
                    "java\\ml6002b2022\\Lab3\\UCIContinuous\\"+ s + "\\"+ s + "_TEST.arff";

            //load the data
            train = DatasetLoading.loadData(trainPath);
            test = DatasetLoading.loadData(testPath);

            //create classifiers
            Bagging bagging = new Bagging();
            RandomForest randomForest = new RandomForest();

            //build the classifier
            bagging.buildClassifier(train);


            //get the accuracy of the classifiers
            double[] acc = new double[21];
            acc[0] = ClassifierTools.accuracy(test, bagging);

            //10 trees
            randomForest.buildClassifier(train);
            acc[1] = ClassifierTools.accuracy(test, randomForest);

            for (int i = 0; i < 19; i++) {
                randomForest = new RandomForest();
                randomForest.setNumTrees(50 + 50*i);
                randomForest.buildClassifier(train);
                acc[2+i] = ClassifierTools.accuracy(test, randomForest);
            }

            accuracies.add(acc);
        }

        System.out.println("Writing to File....");

        try {

            String csvPath = "C:\\Users\\Rls20\\Documents\\GitHub" +
                    "\\tsml\\src\\main\\" +
                    "java\\ml6002b2022\\Lab3\\" +
                    "Averages_" + java.util.UUID.randomUUID() +".csv";
            File csvFile = new File(csvPath);

            if (!csvFile.createNewFile()){
                throw new Exception("ERROR: CSV File already exists with that name");
            }

            FileWriter fileWriter = new FileWriter(csvFile);
            BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);
            PrintWriter writer = new PrintWriter(bufferedWriter);

            StringBuilder sb = new StringBuilder();

            sb.append("bagging Acc");
            sb.append(",");
            sb.append("randFrst10 Acc");
            for (int i = 0; i < 19; i++) {

                sb.append(",");
                sb.append("randFrst" + (50 + 50*i) + " Acc");
            }
            sb.append("\n");

            writer.write(sb.toString());

            for (double[] d:
                    accuracies) {
                sb.setLength(0); //clear the string builder
                for (double dd :
                        d) {
                    sb.append(dd);
                    sb.append(",");
                }
                sb.append("\n");
                writer.write(sb.toString());
            }
            sb.setLength(0);
            sb.append("\n");
            sb.append("bagging Avr Acc");
            sb.append(",");
            sb.append("randFrst10 Avr Acc");
            for (int i = 0; i < 19; i++) {

                sb.append(",");
                sb.append("randFrst" + (50 + 50*i) + " Avr Acc");
            }
            sb.append("\n");

            char c = 'A';
            for (int i = 0; i < accuracies.size(); i++) {
                c+=i;
                sb.append("=AVERAGE(" + c + "2:" + c + "34)");
                sb.append(",");
            }
            sb.append("\n");
            writer.write(sb.toString());

            writer.close();
            System.out.println("Done Writing File");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

    }

    public static void task4() throws Exception {
        String pathTrain = "C:\\Users\\Rls20\\Documents\\GitHub\\tsml" +
                "\\src\\main\\java\\ml6002b2022\\Lab3\\UCIContinuous\\Adiac\\" +
                "Adiac_TRAIN.arff";
        String pathTest = "C:\\Users\\Rls20\\Documents\\GitHub\\tsml" +
                "\\src\\main\\java\\ml6002b2022\\Lab3\\UCIContinuous\\Adiac\\" +
                "Adiac_TEST.arff";

        Instances train, test;

        train = DatasetLoading.loadData(pathTrain);
        test = DatasetLoading.loadData(pathTest);

        // create all the classifiers
        AdaBoostM1 ada= new AdaBoostM1();
        RandomForest randomForest = new RandomForest();
        RandomForest randomForest100 = new RandomForest();
        randomForest100.setNumTrees(100);
        RandomForest randomForest500 = new RandomForest();
        randomForest100.setNumTrees(500);
        LogitBoost logitBoost = new LogitBoost();
        Bagging bagging = new Bagging();
        J48 j48 = new J48();

        //train the classifiers
        ada.buildClassifier(train);
        randomForest.buildClassifier(train);
        randomForest100.buildClassifier(train);
        randomForest500.buildClassifier(train);
        logitBoost.buildClassifier(train);
        bagging.buildClassifier(train);
        j48.buildClassifier(train);

        // put classifiers in an arralist to be able to iterate through using each one more easily
        ArrayList<AbstractClassifier> classifiers = new ArrayList<>();
        classifiers.add(ada);
        classifiers.add(randomForest);
        classifiers.add(randomForest100);
        classifiers.add(randomForest500);
        classifiers.add(logitBoost);
        classifiers.add(bagging);
        classifiers.add(j48);

        //use classifiers on test data
        double[] accur = new double[classifiers.size()];
        for (int i = 0; i < classifiers.size(); i++) {
            System.out.println("Classifer: " + classifiers.get(i).getClass().getSimpleName());;
            accur[i] = ClassifierTools.accuracy(test, classifiers.get(i));
            System.out.println("---Accuracy: " + accur[i]);
        }

}

    public static void main(String[] args) throws Exception {
        //BaggingVsRandSecondExperiment();
        task4();
    }
}
