package ML_Module;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileReader;
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
        }catch(Exception e){
            System.out.println("Exception caught: " + e);
        }
        assert instances != null;
        instances.setClassIndex(instances.numAttributes()-1);
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

        for (Instance i :
                data) {
            classQuantity[i.classValue()]++;
        }

        for (int i = 0; i < classNum; i++) {
            classQuantity[i] = classQuantity[i]/data.numInstances();
        }

        return classQuantity;
    }

    //test helper functions
    public static void main(String[] args) {
        String dataPath = "Arsenal_TRAIN.arff";
        Instances data = loadClassificationData(dataPath);
        System.out.println(data);

        Instances[] split = new Instances[2];
        split = splitData(data, 0.5);
        System.out.println("Test: \n" + split[0]);
        System.out.println("Train: \n" + split[1]);

        double[] classDist = classDistribution(data);
        System.out.println("Class Count: " + Arrays.toString(classDist));
    }
}
