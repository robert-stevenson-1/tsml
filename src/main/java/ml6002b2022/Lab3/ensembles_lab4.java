package ml6002b2022.Lab3;

import core.contracts.Dataset;
import experiments.data.DatasetLoading;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.LogitBoost;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

public class ensembles_lab4 {

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

    public static void logitBoostExample() throws Exception {

        String pth ="C:\\Temp\\";
        Instances train = DatasetLoading.loadData(pth+"DummyProblem.arff");
        System.out.println(train);
        LogitBoost logit = new LogitBoost();
        logit.setDebug(true);

        logit.buildClassifier(train);
    }

    public static void baggingExperiments() throws Exception {
        Instances train,test;
        String path= "C:\\Temp\\UCIContinuous\\";
        System.out.println(" number of problems = "+problems.length);
        double meanDiff=0;
        double meanB=0, meanRF=0;
        int count=0;
        for(String str:problems){
            train = DatasetLoading.loadData(path+str+"\\"+str+"_TRAIN.arff");
            test = DatasetLoading.loadData(path+str+"\\"+str+"_TEST.arff");
            Bagging b= new Bagging();
            b.setNumIterations(500);
            RandomForest rf = new RandomForest();
            rf.setNumTrees(500);
            b.buildClassifier(train);
            rf.buildClassifier(train);
            double accB=ClassifierTools.accuracy(test,b);
            double accRF=ClassifierTools.accuracy(test,rf);
            meanDiff+=accRF-accB;
            meanB+=accB;
            meanRF+=accRF;
            if(accRF>accB)
                count++;
            System.out.println(str+ " Bagging = "+accB+" RandF  = "+accRF);
        }
        meanDiff/=problems.length;
        meanB/=problems.length;
        meanRF/=problems.length;
        System.out.println("RF wins "+count++);
        System.out.println("Mean Diff "+meanDiff);
        System.out.println("Mean B "+meanB);
        System.out.println("Mean RF  "+meanRF);

    }

    public static void boostingExperiments(){
        AdaBoostM1 ada= new AdaBoostM1();
        System.out.println(" Resampling ="+ada.getUseResampling());

    }
    public static void main(String[] args) throws Exception {
        logitBoostExample();
 //       boostingExperiments();
    }
}
