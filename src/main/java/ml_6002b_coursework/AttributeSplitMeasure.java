package ml_6002b_coursework;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Enumeration;

/**
 * Interface for alternative attribute split measures for Part 2.2 of the coursework
 */
public abstract class AttributeSplitMeasure {

    // the learned split value that was used for splitting on numeric (Read-only, is set when splitDataOnNumeric usage)
    private double splitVal = 0;
        // Added based on my interpretation from the guidance given in:
        //   https://learn.uea.ac.uk/webapps/discussionboard/do/message?action=list_messages&course_id=_135859_1&nav=discussion_board&conf_id=_117994_1&forum_id=_14532618_1&message_id=_4585997_1#
        //   from the Blackboard Discussion board question asked in 01/05/2022, and answered on 03/05/2022

    public double getSplitVal() {
        return splitVal;
    }

    public abstract double computeAttributeQuality(Instances data, Attribute att) throws Exception;

    /**
     * Splits a dataset according to the values of a nominal attribute.
     *
     * @param data the data which is to be split
     * @param att the attribute to be used for splitting
     * @return the sets of instances produced by the split
     */
    public Instances[] splitData(Instances data, Attribute att) {
        Instances[] splitData = new Instances[att.numValues()];
        for (int i = 0; i < att.numValues(); i++) {
            splitData[i] = new Instances(data, data.numInstances());
        }

        for (Instance inst: data) {
            splitData[(int) inst.value(att)].add(inst);
        }

        for (Instances split : splitData) {
            split.compactify();
        }

        return splitData;
    }

    /**
     * Splits a dataset according to the values of a numerical attribute.
     *
     * @param data the data which is to be split
     * @param att the attribute to be used for splitting
     * @param splitVal the value we split on, is 0.0 then calculate 'splitVal'
     * @return the sets of instances produced by the split
     */
    public Instances[] splitDataOnNumeric(Instances data, Attribute att, double splitVal) throws Exception {
        if (att.isNumeric()){
            Instances[] splitData = new Instances[2]; // doing a binary split so only 2 split sets
            // initialise the size of these instances
            splitData[0] = new Instances(data, data.numInstances());
            splitData[1] = new Instances(data, data.numInstances());

            // no splitVal specified so calculate the best value
            //TODO: Try Reduction on variance: https://www.analyticsvidhya.com/blog/2020/06/4-ways-split-decision-tree/
            if (splitVal == 0){ // calc the meanattSplitMeasure = {IGAttributeSplitMeasure@791}
/*
            This loop to calc the mean doesn't calc the same as 
                for (Instance inst :
                        data) {
                    this.splitVal += inst.value(att);
                }
                this.splitVal /= data.numInstances();*/
                this.splitVal = data.meanOrMode(att);
            }else {
                this.splitVal =splitVal;
            }

            //split the data on that value
            for (Instance inst: data) {
                if (inst.value(att) < this.splitVal){
                    splitData[0].add(inst);
                }else {
                    splitData[1].add(inst);
                }
            }

            // Reduce the set of instances sizes by decreasing the capacity of the set
            //   equal to the number of instances inside them.
            for (Instances split : splitData) {
                split.compactify();
            }

            //return the split data
            return splitData;
        }else{
            throw new Exception("Attribute is not NUMERIC. (Att: " + att + ")");
        }
    }
}
