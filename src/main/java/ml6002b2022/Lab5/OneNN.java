package ml6002b2022.Lab5;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class OneNN extends AbstractClassifier {
    private Instances trainData;
    @Override
    public void buildClassifier(Instances data) throws Exception {
        trainData = data;
    }
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        Instance closestInst = trainData.firstInstance();
        double dist, closestDist = Double.POSITIVE_INFINITY;
        for (Instance inst :
            trainData){
            dist = distance(inst, closestInst);
            if (dist < closestDist){
                closestDist = dist;
                closestInst = inst;
            }
        }
        return closestInst.classValue();
    }

    double distance(Instance instance, Instance instance2){
        double sum = 0;
        for (int i = 0; i < trainData.numAttributes() - 1; i++) {
            sum += Math.pow(instance.value(i) - instance2.value(i), 2);
        }
        return Math.sqrt(sum);
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] predProb = new double[trainData.numClasses()];
        double pred = classifyInstance(instance);

        predProb[(int) pred] = 1.0;

        return predProb;
    }
}
