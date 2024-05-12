package ml_6002b_coursework;

import utilities.ClassifierTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import experiments.data.*;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;

public class TreeEnsemble extends AbstractClassifier {

   ArrayList<Classifier> ensemble;
   int numOfTrees = 50;
   long seed = 0;
   double proportionOfAttributes = 0.5;
   boolean averageDistributions = false;

   HashMap<Classifier, Instances> usedAttri = new HashMap<>();

   @Override
   public void buildClassifier(Instances data) throws Exception {
      ensemble = new ArrayList<>();
      //populate the ensemble classifier list
      for (int i = 0; i < numOfTrees; i++) {
         CourseworkTree classifier = new CourseworkTree();
         // randomise the tree's parameters:
         //    Get a random enum value for the attribute measure used for this classifier for the split option
         CourseworkTree.SPLIT_OPTIONS val =
                 CourseworkTree.SPLIT_OPTIONS.values()[(new Random()).nextInt(CourseworkTree.SPLIT_OPTIONS.values().length)];
         classifier.setOptions(val);

         ensemble.add(classifier);
      }

      Random rand;

      //randomly split the data into subsets to train the classifers
      for (Classifier c :
              ensemble) {
         //shuffle the data
         if (this.seed != 0){ //check if the ensemble has a seed set
            // now seed to use a random seed
            data.randomize(new Random());
         }else {
            //seed set so use this for setting the random number generator
            data.randomize(new Random(this.seed));
         }
         //create the Random subset
         Instances subset = new Instances(data, 0, (int)(data.numInstances()*proportionOfAttributes));
         usedAttri.put(c, subset); // store the used subset for this classifier.
         c.buildClassifier(subset); //train the classifier with the subset
      }
   }

   @Override
   public double classifyInstance(Instance instance) throws Exception {
      int[] counts=new int[instance.numClasses()];
      for(Classifier c:ensemble){
         counts[(int)c.classifyInstance(instance)]++;
      }
      int argMax=0;
      //get the index of the final Vote that's highest
      for(int i=1;i<counts.length;i++)
         if(counts[i]>counts[argMax])
            argMax=i;
      return argMax;
   }

   @Override
   public double[] distributionForInstance(Instance instance) throws Exception {
      //store the probability or proportion values for the class distribution
      double[] pro = new double[instance.numClasses()];

      // get the avr. prob, for the distribution of the classifiers
      if (averageDistributions){
         for(Classifier c:ensemble){
            double[] d = c.distributionForInstance(instance);
            for(int i=0;i<d.length;i++)
               pro[i]+=d[i];
         }

      // get the proportion of the votes for each class
      }else{
         for (Classifier c :
                 ensemble) {
            pro[(int) c.classifyInstance(instance)]++;
         }
      }

      for(int i=0;i<pro.length;i++)
         pro[i]/=numOfTrees;
      return pro;
   }

   //===================================================
   //===============Accessor and Getters================
   //===================================================


   public boolean isAverageDistributions() {
      return averageDistributions;
   }

   public void setAverageDistributions(boolean averageDistributions) {
      this.averageDistributions = averageDistributions;
   }

   public int getNumOfTrees() {
      return numOfTrees;
   }

   public void setNumOfTrees(int numOfTrees) {
      this.numOfTrees = numOfTrees;
   }

   public long getSeed() {
      return seed;
   }

   public void setSeed(long seed) {
      this.seed = seed;
   }

   public double getProportionOfAttributes() {
      return proportionOfAttributes;
   }

   public void setProportionOfAttributes(double proportionOfAttributes) {
      this.proportionOfAttributes = proportionOfAttributes;
   }

   public static void main(String[] args) throws Exception {

      System.out.println(
              "===============================================\n" +
              "=================optDigit Data=================\n" +
              "===============================================\n");

      Instances optDigitsData = DatasetLoading.loadData("src/main/java/ml_6002b_coursework/test_data/optdigits.arff");

      //split the data into a training and a test set of data
      Instances[] splitData = WekaTools.splitData(optDigitsData, 0.3); //70% train, 30% test

      //make and train the classifier
      TreeEnsemble ensemble = new TreeEnsemble();
      ensemble.buildClassifier(splitData[0]);

      System.out.println(splitData[0].numInstances());
      System.out.println(splitData[1].numInstances());

      System.out.println("Tree Ensemble accuracy: " + ClassifierTools.accuracy(splitData[1], ensemble));

      for (int i = 0; i < 5; i++) {
         double[] distProb = ensemble.distributionForInstance(splitData[1].get(i));
         System.out.println("Probability estimate for test case <" +
                 i +"> = " + Arrays.toString(distProb));
      }

      System.out.println(
             "===============================================\n" +
             "================Chinatown Data=================\n" +
             "===============================================\n");

      Instances chinatownData = DatasetLoading.loadData("src/main/java/ml_6002b_coursework/test_data/Chinatown.arff");

      //split the data into a training and a test set of data
      Instances[] splitDataChina = WekaTools.splitData(chinatownData, 0.3); //70% train, 30% test

      //make and train the classifier
      ensemble = new TreeEnsemble();
      ensemble.buildClassifier(splitDataChina[0]);

      System.out.println(splitDataChina[0].numInstances());
      System.out.println(splitDataChina[1].numInstances());

      System.out.println("Tree Ensemble accuracy: " + ClassifierTools.accuracy(splitDataChina[1], ensemble));

      for (int i = 0; i < 5; i++) {
         double[] distProb = ensemble.distributionForInstance(splitDataChina[1].get(i));
         System.out.println("Probability estimate for test case <" +
                 i +"> = " + Arrays.toString(distProb));
      }

   }
}
