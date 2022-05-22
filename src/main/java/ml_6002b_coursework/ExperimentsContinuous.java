package ml_6002b_coursework;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Instances;

import experiments.data.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;

public class ExperimentsContinuous {

   public static void getDataTableInfo(Instances train_Data, Instances test_Data) throws Exception {
      Instances both = new Instances(test_Data);
      both.addAll(train_Data);

      System.out.println("Num of attributes (train): " + train_Data.numAttributes());
      System.out.println("Num of attributes (test): " + test_Data.numAttributes());
      System.out.println("Num of (Both) cases: " + both.numInstances());
      System.out.println("Num of (train) cases: " + train_Data.numInstances());
      System.out.println("Num of (test) cases: " + test_Data.numInstances());
      System.out.println("Num of classes(Both): " +both.numClasses());
      System.out.println("Num of classes(train): " + train_Data.numClasses());
      System.out.println("Num of classes(test): " + test_Data.numClasses());
      System.out.println("Class distribution (Both): " + Arrays.toString(WekaTools.classDistribution(both)));
      System.out.println("Class distribution (train): " + Arrays.toString(WekaTools.classDistribution(train_Data)));
      System.out.println("Class distribution (test): " + Arrays.toString(WekaTools.classDistribution(test_Data)));
   }

   public static String comparisonTest(Instances data_TRAIN, Instances data_TEST) throws Exception {
      long seed = 10;
      //store result data
      StringBuilder result = new StringBuilder();
      ArrayList<Classifier> classifiers = new ArrayList<>();

      CourseworkTree cwTreeIG = new CourseworkTree();
      CourseworkTree cwTreeGini = new CourseworkTree();
      cwTreeGini.setOptions(CourseworkTree.SPLIT_OPTIONS.GINI);
      CourseworkTree cwTreeChi = new CourseworkTree();
      cwTreeChi.setOptions(CourseworkTree.SPLIT_OPTIONS.CHI_SQUARED);
      TreeEnsemble treeEnsemble10 = new TreeEnsemble();
      treeEnsemble10.setNumOfTrees(10);
      TreeEnsemble treeEnsemble50 = new TreeEnsemble();
      treeEnsemble50.setNumOfTrees(50);
      TreeEnsemble treeEnsemble100 = new TreeEnsemble();
      treeEnsemble100.setNumOfTrees(100);
      TreeEnsemble treeEnsemble250 = new TreeEnsemble();
      treeEnsemble250.setNumOfTrees(250);
      TreeEnsemble treeEnsemble500 = new TreeEnsemble();
      treeEnsemble500.setNumOfTrees(500);
      TreeEnsemble treeEnsemble750 = new TreeEnsemble();
      treeEnsemble750.setNumOfTrees(750);
      TreeEnsemble treeEnsemble1000 = new TreeEnsemble();
      treeEnsemble1000.setNumOfTrees(1000);


      RandomTree randomTree = new RandomTree();

      RandomForest randomForest = new RandomForest();
      RandomForest randomForest50 = new RandomForest();
      randomForest50.setNumTrees(50);
      RandomForest randomForest100 = new RandomForest();
      randomForest100.setNumTrees(100);
      RandomForest randomForest250 = new RandomForest();
      randomForest250.setNumTrees(250);
      RandomForest randomForest500 = new RandomForest();
      randomForest500.setNumTrees(500);
      RandomForest randomForest750 = new RandomForest();
      randomForest750.setNumTrees(750);
      RandomForest randomForest1000 = new RandomForest();
      randomForest1000.setNumTrees(1000);

      RotationForest rotationForest = new RotationForest();
      RotationForest rotationForest10 = new RotationForest();
      rotationForest10.setNumberOfGroups(true);
      rotationForest10.setMinGroup(10);
      rotationForest10.setMaxGroup(10);
      RotationForest rotationForest50 = new RotationForest();
      rotationForest50.setNumberOfGroups(true);
      rotationForest50.setMinGroup(50);
      rotationForest50.setMaxGroup(50);
      RotationForest rotationForest100 = new RotationForest();
      rotationForest100.setNumberOfGroups(true);
      rotationForest100.setMinGroup(100);
      rotationForest100.setMaxGroup(100);
      RotationForest rotationForest250 = new RotationForest();
      rotationForest250.setNumberOfGroups(true);
      rotationForest250.setMinGroup(250);
      rotationForest250.setMaxGroup(250);
      RotationForest rotationForest500 = new RotationForest();
      rotationForest500.setNumberOfGroups(true);
      rotationForest500.setMinGroup(500);
      rotationForest500.setMaxGroup(500);
      RotationForest rotationForest750 = new RotationForest();
      rotationForest750.setNumberOfGroups(true);
      rotationForest750.setMinGroup(750);
      rotationForest750.setMaxGroup(750);
      RotationForest rotationForest1000 = new RotationForest();
      rotationForest1000.setNumberOfGroups(true);
      rotationForest1000.setMinGroup(1000);
      rotationForest1000.setMaxGroup(1000);


      J48 j48 = new J48();
      SMO svm = new SMO();

      //cwTree.setSeed();
     //treeEnsemble.seed = seed;
     //randomTree.setSeed(seed);
     //j48.setSeed(seed);

      //add to arraylist to iterate through classifiers
      classifiers.add(cwTreeIG);
      classifiers.add(cwTreeGini);
      classifiers.add(cwTreeChi);

      classifiers.add(treeEnsemble10);
      classifiers.add(treeEnsemble50);
      classifiers.add(treeEnsemble100);
      classifiers.add(treeEnsemble250);

      classifiers.add(randomTree);
      classifiers.add(randomForest);
      classifiers.add(randomForest50);
      classifiers.add(randomForest100);
      classifiers.add(randomForest250);
      classifiers.add(randomForest500);
      classifiers.add(randomForest750);
      classifiers.add(randomForest1000);

      classifiers.add(rotationForest);
      classifiers.add(rotationForest50);
      classifiers.add(rotationForest100);
      classifiers.add(rotationForest250);
      classifiers.add(rotationForest500);
      classifiers.add(rotationForest750);
      classifiers.add(rotationForest1000);

      classifiers.add(j48);
      classifiers.add(svm);

      //train the classifiers
/*      cwTree.buildClassifier(data_TRAIN);
      treeEnsemble.buildClassifier(data_TRAIN);
      randomTree.buildClassifier(data_TRAIN);
      randomForest.buildClassifier(data_TRAIN);
      rotationForest.buildClassifier(data_TRAIN);
      j48.buildClassifier(data_TRAIN);
      svm.buildClassifier(data_TRAIN);*/

      //create evaluator
      Evaluation evaluation = new Evaluation(data_TRAIN);

      //Get the accuracies
      for (Classifier c :
              classifiers) {
         long startTime = System.nanoTime();
         c.buildClassifier(data_TRAIN);
         long endTime = System.nanoTime();
         long duration = (endTime - startTime)/1000000; //build time in millisecs

         String classifierName = c.getClass().getSimpleName();
         evaluation.evaluateModel(c, data_TEST);
         double accuracy = 1-evaluation.errorRate();
         double precision = evaluation.weightedPrecision();
         double recall = evaluation.weightedRecall();
         double fMeasure = evaluation.weightedFMeasure();
         double areaUnderROC = evaluation.weightedAreaUnderROC();

         result.append(classifierName).append(",");
         result.append(accuracy).append(",");
         result.append(precision).append(",");
         result.append(recall).append(",");
         result.append(fMeasure).append(",");
         result.append(areaUnderROC).append(",");
         result.append(duration).append(",");
         result.append("\n");
      }
      //result.append("\n");
      return result+"\n";
   }

   protected static void save(Instances data, String filename) throws Exception {
      BufferedWriter  writer;

      writer = new BufferedWriter(new FileWriter(filename));
      writer.write(data.toString());
      writer.newLine();
      writer.flush();
      writer.close();
   }

   public static void main(String[] args) throws Exception {

      //create result file
      File resultsDTree = new File(
              "src/main/java/ml_6002b_coursework/results/" +
                      "ChlorineConcentration_DTrees.csv");

      FileWriter fileWriter = new FileWriter(resultsDTree, true);

      fileWriter.write("========================\n");
      fileWriter.write("Classifier," +
              "Accuracy," +
              "Weighted Precision," +
              "Weighted Recall," +
              "Weighted F-Measure," +
              "Weighted Area under ROC," +
              "build (Train) time (ms)," +
              "\n");

/*      for (String problem :
              DatasetLists.continuousAttributeProblems) {
         Instances problemData = DatasetLoading.loadData(
                 "src/main/java/ml_6002b_coursework/test_data/UCI Continuous/" +
                         problem + "/" + problem + ".arff");
         System.out.println("Problem: " + problem);

//         String result = problem + "," +
//                 dTreeComp(problemData);
         String result = problem + "," +
                 dTreeComp(problemData);
         System.out.println(result);
         break;

         //fileWriter.write(result);
      }
      */
      String problem = "ChlorineConcentration";

      //load problem data
      Instances problemData_TRAIN = DatasetLoading.loadData(
              "src/main/java/ml_6002b_coursework/test_data/ChlorineConcentration/"+problem+"_TRAIN.arff");
      Instances problemData_TEST = DatasetLoading.loadData(
              "src/main/java/ml_6002b_coursework/test_data/ChlorineConcentration/"+problem+"_TEST.arff");
      System.out.println("Problem: " + problem);

      //CONVERT TO NOMINAL DATA FOR COMPARISON WITH NUMERIC ATTRIBUTES
      // METHOD SOURCE: https://waikato.github.io/weka-wiki/discretizing_datasets/
      // NumericToNominal was an alternative tryed but resulted in errors
      Discretize toNominal = new Discretize();

      //convert the TEST data
      toNominal.setInputFormat(problemData_TEST);
      Instances nomProblemData_TEST = Filter.useFilter(problemData_TEST, toNominal);

      //convert the TRAIN data
      toNominal.setInputFormat(problemData_TRAIN);
      Instances nomProblemData_TRAIN = Filter.useFilter(problemData_TRAIN, toNominal);

      save(nomProblemData_TEST,"src/main/java/ml_6002b_coursework/test_data/ChlorineConcentration/" +
              "nomProblemData_TEST.arff");
      save(nomProblemData_TRAIN,"src/main/java/ml_6002b_coursework/test_data/ChlorineConcentration/" +
              "nomProblemData_TRAIN.arff");

//         String result = problem + "," +
//                 dTreeComp(problemData);
      getDataTableInfo(problemData_TRAIN, problemData_TEST);

      String result = comparisonTest(problemData_TRAIN, problemData_TEST);
      System.out.println("====(Numeric Attr) Comparison Run===");
      System.out.println(result);
      fileWriter.write("====(Numeric Attr) Comparison Run===\n");
      fileWriter.write(result);


      String nomResult = comparisonTest(nomProblemData_TRAIN, nomProblemData_TEST);
      System.out.println("====(Nominal Attr) Comparison Run===");
      System.out.println(nomResult);
      fileWriter.write("====(Nominal Attr) Comparison Run===\n");
      fileWriter.write(nomResult);


      fileWriter.close();
      System.out.println("Done Writing results");
   }

}
