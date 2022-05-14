package ml_6002b_coursework;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import experiments.data.*;

import java.util.Arrays;

public class IGAttributeSplitMeasure extends AttributeSplitMeasure {
    private boolean useGain;

    public IGAttributeSplitMeasure() {
        this.useGain = true; // default use Information Gain
    }

    public IGAttributeSplitMeasure(boolean useGain) {
        this.useGain = useGain;
    }

    @Override
    public double computeAttributeQuality(Instances data, Attribute att) throws Exception {

        //shape our contingency table
        //  Rows: all possible unique values for the attibute 'att'
        //  col: the num of observable values for that possible attribute value of 'att'
        int[][] conTable = new int[att.numValues()][data.numClasses()];

        if (att.isNumeric()){ //handle numeric attribute types
            Instances[] splitData;
            // split the data prior to computing the quality
            splitData = splitDataOnNumeric(data, att,0);
            // re-shape the contingency table to have 2 row as the split data result in
            //   simplifying the attribute values into 2 sets
            conTable = new int[2][data.numClasses()];
            //create the contingency table
            for (int i = 0; i < 2; i++) {
                // take the data instances and format them into a contingency table
                for (Instance instance: splitData[i]) {
                    //tally up the observed instances for this attribute value
                    conTable[i][(int) instance.classValue()]++;
                }
            }
        }else {
            // take the data instances and format them into a contingency table
            for (Instance instance: data) {
                //tally up the observed instances for this attribute value
                conTable[(int) instance.value(att.index())][(int) instance.classValue()]++;
            }
        }

//        // take the data instances and format them into a contigency table
//        for (Instance instance: data) {
//            //tally up the observed instances for this attribute value
//            conTable[(int) instance.value(att.index())][(int) instance.classValue()]++;
//        }

//        System.out.println("=============");
//        System.out.println("Con table: \n" + Arrays.deepToString(conTable).replace("], ", "]\n") + "\n");
//        System.out.println("=============");

        // decide if we return the InfoGain or the InfoGainRatio
        if (useGain){
            //return info gain
            return AttributeMeasures.measureInformationGain(conTable);
        }else{
            //return infoGainRation
            return AttributeMeasures.measureInformationGainRatio(conTable);
        }
    }

    public boolean isUseGain() {
        return useGain;
    }

    public void setUseGain(boolean useGain) {
        this.useGain = useGain;
    }

    /**
     * Main method.
     *
     * @param args the options for the split measure main
     */
    public static void main(String[] args) throws Exception {
        // load whisky data for test
        Instances whiskyData = DatasetLoading.loadData("src/main/java/ml_6002b_coursework/test_data/Whisky.arff");

        IGAttributeSplitMeasure igMeasure= new IGAttributeSplitMeasure();

        System.out.println("========================================");
        System.out.println("Data: \n" + whiskyData);


        System.out.println("========================================");
        System.out.println("==========Use Information Gain==========");
        System.out.println("========================================");
        System.out.println("measure <measureInformationGain> for attribute <Peaty> splitting diagnosis = "
                + igMeasure.computeAttributeQuality(whiskyData, whiskyData.attribute(0)));

        System.out.println("measure <measureInformationGain> for attribute <Woody> splitting diagnosis = "
                + igMeasure.computeAttributeQuality(whiskyData, whiskyData.attribute(1)));

        System.out.println("measure <measureInformationGain> for attribute <Sweet> splitting diagnosis = "
                + igMeasure.computeAttributeQuality(whiskyData, whiskyData.attribute(2)));



        System.out.println("========================================");
        System.out.println("=======Use Information Gain Ratio=======");
        System.out.println("========================================");

        igMeasure.setUseGain(false);

        System.out.println("measure <measureInformationGainRatio> for attribute <Peaty> splitting diagnosis = "
                + igMeasure.computeAttributeQuality(whiskyData, whiskyData.attribute(0)));

        System.out.println("measure <measureInformationGainRatio> for attribute <Woody> splitting diagnosis = "
                + igMeasure.computeAttributeQuality(whiskyData, whiskyData.attribute(1)));

        System.out.println("measure <measureInformationGainRatio> for attribute <Sweet> splitting diagnosis = "
                + igMeasure.computeAttributeQuality(whiskyData, whiskyData.attribute(2)));
    }

}
