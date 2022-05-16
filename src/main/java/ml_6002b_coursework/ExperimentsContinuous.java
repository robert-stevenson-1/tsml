package ml_6002b_coursework;

import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Instances;

import experiments.data.*;

import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;

public class ExperimentsContinuous {

   /*
   Decision Trees: Test whether there is any difference in average accuracy for the attribute
   selection methods on the classification problems we have provided. Compare your versions
   of DecisionTree to the Weka ID3 and J48 classifiers.
   */
   public static String dTreeComp(Instances data) throws Exception {
      //store result data

      StringBuilder result = new StringBuilder();
      ArrayList<Classifier> classifiers = new ArrayList<>();

      CourseworkTree cwTree = new CourseworkTree();
      TreeEnsemble treeEnsemble = new TreeEnsemble();
      RandomTree randomTree = new RandomTree();
      J48 j48 = new J48();

      //add to arraylist to iterate through classifiers
      classifiers.add(cwTree);
      classifiers.add(treeEnsemble);
      classifiers.add(randomTree);
      classifiers.add(j48);

      //split data
      Instances[] split = new Instances[2];
      split = WekaTools.splitData(data, 0.4); // 60% train, 40% test

      //train the classifiers
      cwTree.buildClassifier(split[0]);
      treeEnsemble.buildClassifier(split[0]);
      randomTree.buildClassifier(split[0]);
      j48.buildClassifier(split[0]);

      //Get the accuracies
      for (Classifier c :
              classifiers) {
         String classifierName = c.getClass().getSimpleName();
         double accuracy = ClassifierTools.accuracy(split[1], c);
         result.append(accuracy).append(",");
      }
      result.append("\n");
      return result.toString();
   }

   public static String ensembleComp(Instances data) throws Exception {
      //store result data
      StringBuilder result = new StringBuilder();
      ArrayList<Classifier> classifiers = new ArrayList<>();

      TreeEnsemble treeEnsemble = new TreeEnsemble();
      RandomForest randomForest = new RandomForest();

      //add to arraylist to iterate through classifiers
      classifiers.add(treeEnsemble);
      classifiers.add(randomForest);

      //split data
      Instances[] split = new Instances[2];
      split = WekaTools.splitData(data, 0.4); // 60% train, 40% test

      //train the classifiers
      treeEnsemble.buildClassifier(split[0]);
      randomForest.buildClassifier(split[0]);

      //Get the accuracies
      for (Classifier c :
              classifiers) {
         double accuracy = ClassifierTools.accuracy(split[1], c);
         result.append(accuracy).append(",");
      }
      result.append("\n");
      return result.toString();
   }


   public static void main(String[] args) throws Exception {

      //create result file
      File resultsDTree = new File(
              "src/main/java/ml_6002b_coursework/results/" +
                      "continous_DTrees.txt");

      FileWriter fileWriter = new FileWriter(resultsDTree, true);

      fileWriter.write("Problem," +
              "CourseworkTree Accuracy," +
              "TreeEnsemble Accuracy," +
              "RandomTree Accuracy," +
              "J48 Accuracy"+
              "\n");
      //load problem data
      for (String problem :
              DatasetLists.continuousAttributeProblems) {
         Instances problemData = DatasetLoading.loadData(
                 "src/main/java/ml_6002b_coursework/test_data/UCI Continuous/" +
                         problem + "/" + problem + ".arff");
         System.out.println("Problem: " + problem);

//         String result = problem + "," +
//                 dTreeComp(problemData);
         String result = problem + "," +
                 dTreeComp(problemData);

         fileWriter.write(result);
      }
      fileWriter.close();
   }

}
