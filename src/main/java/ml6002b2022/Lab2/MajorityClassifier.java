package ml6002b2022.Lab2;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;

import static ml6002b2022.Lab2.WekaTools.classDistribution;
import static ml6002b2022.Lab2.WekaTools.loadClassificationData;

public class MajorityClassifier extends AbstractClassifier {

    int mostCommomClass;

    public MajorityClassifier() {

    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        double[] dist = classDistribution(data);
        System.out.println(Arrays.toString(dist));

        mostCommomClass = getIndexOfLargest(dist);
        System.out.println("Largest Dist Value: " + dist[mostCommomClass]);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return super.classifyInstance(instance);
    }

    private int getIndexOfLargest(double[] array )
    {
        if ( array == null || array.length == 0 ) return -1; // null or empty

        int largest = 0;
        for ( int i = 1; i < array.length; i++ )
        {
            if ( array[i] > array[largest] ) largest = i;
        }
        return largest; // position of the first largest found
    }

    public static void main(String[] args) throws Exception {
        Instances data = loadClassificationData("src/main/java/ml6002b2022/Lab2/Arsenal_TEST.arff");
        //Instances data = loadClassificationData("src/main/java/ml6002b2022/Lab2/Arsenal_TRAIN.arff");
        MajorityClassifier c = new MajorityClassifier();
        c.buildClassifier(data);

    }
}