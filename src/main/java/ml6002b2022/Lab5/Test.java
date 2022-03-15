package ml6002b2022.Lab5;

import experiments.data.DatasetLoading;
import weka.core.Instance;
import weka.core.Instances;

import java.io.IOException;
import java.lang.reflect.Array;
import java.util.Arrays;

public class Test {
    public static void main(String[] args) throws Exception {
        Instances train, test;
        train = DatasetLoading.loadData("C:\\Users\\Rls20\\Documents\\GitHub\\tsml\\src" +
                "\\main\\java\\ml6002b2022\\Lab5\\Arsenal_TRAIN.arff");
        test = DatasetLoading.loadData("C:\\Users\\Rls20\\Documents\\GitHub\\tsml\\src" +
                "\\main\\java\\ml6002b2022\\Lab5\\Arsenal_TEST.arff");

        OneNN oneNN = new OneNN();
        oneNN.buildClassifier(train);

        for (Instance inst :
                test) {
            System.out.println("Predicted Class: " + oneNN.classifyInstance(inst) +
                    " Actual: " + inst.classValue());
        }

        double[] distribution = oneNN.distributionForInstance(test.instance(0));
        System.out.println("Predicted (probabilty) (inst[0]): " + Arrays.toString(distribution));
    }
}
