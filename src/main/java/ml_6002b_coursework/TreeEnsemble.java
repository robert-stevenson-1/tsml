package ml_6002b_coursework;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Random;

public class TreeEnsemble extends AbstractClassifier {

   ArrayList<Classifier> ensemble;
   int numOfTrees = 50;
   long seed = 0;
   double proportionOfAttributes = 0.5;

   @Override
   public void buildClassifier(Instances data) throws Exception {
      ensemble = new ArrayList<>();
      //populate the ensemble classifier list
      for (int i = 0; i < numOfTrees; i++) {
         Classifier classifier = new CourseworkTree();
         // randomise the tree's parameters:
         //    Get a random enum value for the attribute measure used for this classifier for the split option
         CourseworkTree.SPLIT_OPTIONS val =
                 CourseworkTree.SPLIT_OPTIONS.values()[(new Random()).nextInt(CourseworkTree.SPLIT_OPTIONS.values().length)];
         ((CourseworkTree)classifier).setOptions(val);

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
      for(int i=1;i<counts.length;i++)
         if(counts[i]>counts[argMax])
            argMax=i;
      return argMax;
   }

   @Override
   public double[] distributionForInstance(Instance instance) throws Exception {
      double[] probs = new double[instance.numClasses()];

      for(Classifier c:ensemble){
         double[] d = c.distributionForInstance(instance);
         for(int i=0;i<d.length;i++)
            probs[i]+=d[i];
      }

      double sum=0;

      for (double prob : probs) sum += prob;

      for(int i=0;i<probs.length;i++)
         probs[i]/=sum;
      return probs;
   }

   //===================================================
   //===============Accessor and Getters================
   //===================================================


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
}
