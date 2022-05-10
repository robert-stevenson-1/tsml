package ml_6002b_coursework;

import org.checkerframework.checker.units.qual.A;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * Empty class for Part 2.1 of the coursework.
 */
public class AttributeMeasures {

    public static double measureInformationGain(int[][] table){
        // Table:
        //                 | ObsVal1    | ObsVal2 |   ...   | ObsValX
        //       -----------------------------------------------------
        //       AttriVal1 |            |         |   ...   |
        //       AttriVal2 |            |         |   ...   |
        //          ...    |     ...    |   ...   |   ...   |   ...
        //       AttriValX |            |         |   ...   |

        double infoGain = 0.0;
        ArrayList<Integer> attriValTotals = new ArrayList<>();
        ArrayList<Integer> obsClassTotals = new ArrayList<>();
        ArrayList<Double> entropy = new ArrayList<>(); // array of double
        double total = 0;

        //calc the total and the attributes value totals
        for (int[] attriVal :
                table) {
            //sum the observed values for the current attriVal and add to the total tally
            int attriValTotal = Arrays.stream(attriVal).sum();
            attriValTotals.add(attriValTotal);
            total += attriValTotal;
        }
        System.out.println("Total: " + total);

        // calc entropy for each attribute
        for (int a = 0; a < table.length; a++){ // loop through each different attribute value
            double tempEntropy = 0.0;
            //calc the Entropy for each attribute value's class values
            for (int i = 0; i < table[a].length; i++) {
                tempEntropy += (table[a][i]/(double)attriValTotals.get(a))*log2Ent(table[a][i]/(double)attriValTotals.get(a));
            }
            tempEntropy *= -1;
            entropy.add(tempEntropy);
        }

        // get the observation scores totals
        for (int i = 0; i <table[0].length; i++) {
            int tempClassTotal = 0;
            for (int j = 0; j < table.length; j++) {
                tempClassTotal += table[j][i];
            }
            obsClassTotals.add(tempClassTotal);
        }

        System.out.println("AttriValTotals: " + attriValTotals);
        System.out.println("ObsScoreTotal: " + obsClassTotals);
        System.out.println("Entropy: " + entropy);

        //get parent entropy and add it to the information gain ready to subtract from
        for (int i = 0; i < obsClassTotals.size(); i++) {
            infoGain += (obsClassTotals.get(i)/total)*log2Ent(obsClassTotals.get(i)/total);
        }
        infoGain *= -1;
        System.out.println("Entropy of Parent: " + infoGain);

        //calc the information gain
        for (int i = 0; i < entropy.size(); i++) {
            infoGain -= ((attriValTotals.get(i)/total)*entropy.get(i));
        }

        return infoGain;
    }

    public static double measureInformationGainRatio(int[][] table){
        double gain = measureInformationGain(table);
        double splitInfo = 0;

        ArrayList<Integer> attriValTotals = new ArrayList<>();
        double total = 0;
        //calc the total and the attributes value totals
        for (int[] attriVal :
                table) {
            //sum the observed values for the current attriVal and store, add it to the total tally
            int attriValTotal = Arrays.stream(attriVal).sum();
            attriValTotals.add(attriValTotal);
            total += attriValTotal;
        }

        for (int a :
                attriValTotals) {
            splitInfo += ((a/total)*log2Ent((a/total)));
        }
        splitInfo *= -1;
        System.out.println("Split Info: " + splitInfo);
        return gain/splitInfo;
    }

    public static double measureGini(int[][] table){
        double gini = 1;
        double total = 0;
        ArrayList<Integer> attriValTotals = new ArrayList<>();
        ArrayList<Integer> obsClassTotals = new ArrayList<>();

        //calc the total
        for (int[] attriVal :
                table) {
            //sum the observed values for the current attriVal and add to the total tally
            int attriValTotal = Arrays.stream(attriVal).sum();
            attriValTotals.add(attriValTotal);
            total += attriValTotal;
        }

        // get the totals observation scores and
        for (int i = 0; i <table[0].length; i++) {
            int tempClassTotal = 0;
            for (int j = 0; j < table.length; j++) {
                tempClassTotal += table[j][i];
            }
            obsClassTotals.add(tempClassTotal);
        }

        //get parent gini impurity and it the gini val ready to subtract from
        for (int i = 0; i < obsClassTotals.size(); i++) {
            gini -= Math.pow(obsClassTotals.get(i)/total, 2);
        }
        //System.out.println("Parent gini purity: " + gini);

        //calc the gini impurity
        for (int i = 0; i < table.length; i++) {
            double attriGini = 1.0;
            // get the gini impurity measure for that attribute value
            for (int j = 0; j < table[0].length; j++) {
                attriGini -= (Math.pow(table[i][j]/(double)attriValTotals.get(i), 2));
            }
            //subtract the gini of the current attribute value from the gini measure that start equal to
            // the parent val and is to be returned later on
            gini -= ((attriValTotals.get(i)/total)*attriGini);
        }

        return gini;
    }

    public static double measureChiSquared(int[][] table){
        ArrayList<Integer> attriValTotals = new ArrayList<>(); // totals for each row
        ArrayList<Integer> obsClassTotals = new ArrayList<>(); // totals for each column
        ArrayList<Double> globalProb = new ArrayList<>(); // global probability for each class value
        double total = 0; // total num of cases
        double chiSquare = 0; // return metric

        //calc the total and the attributes value totals
        for (int[] attriVal :
                table) {
            //sum the observed values for the current attriVal and store, add it to the total tally
            int attriValTotal = Arrays.stream(attriVal).sum();
            attriValTotals.add(attriValTotal);
            total += attriValTotal;
        }

        // get the observation scores totals and the global probabilities
        for (int i = 0; i <table[0].length; i++) {
            int tempClassTotal = 0;
            for (int j = 0; j < table.length; j++) {
                tempClassTotal += table[j][i];
            }
            obsClassTotals.add(tempClassTotal);
            globalProb.add(tempClassTotal/total);
        }
        System.out.println("ObsScoreTotal: " + obsClassTotals);
        System.out.println("global Prob: " + globalProb);

        //calculate the chi-squared statistic
        for (int i = 0; i < table.length; i++) { // for every attribute value in the contingency table
            for (int j = 0; j < table[0].length; j++) {
                // for every value of the attriVal, subtract the expected value from the observed value
                // and square the result. Then divide by the expected value and add the result to
                // the chi-squared statistic
                double expected = attriValTotals.get(i)*globalProb.get(j); //calc the expect val of the is cell of
                System.out.println("Table ["+i+"]["+j+"] expected value: " + expected);
                // the contingency table
                chiSquare += (Math.pow(table[i][j] - expected, 2)/expected);
            }
        }

        return chiSquare;
    }

    //wrapper for the log2 function that will make log2(a) = 0
    private static double log2Ent(double a){
        if (a == 0) return 0;
        return log2(a);
    }

    //implement a log2 function
    private static double log2(double a){
        // prevent illegal values being entered
        if (a <= 0) throw new IllegalArgumentException("a must be positive (a: " + a +")");
        // calc and return the Log base 2 of 'a'
        return Math.log(a) / Math.log(2);
    }

    /**
     * Main method.
     *
     * @param args the options for the attribute measure main
     */
    public static void main(String[] args) {
        System.out.println("Not Implemented.");

/*        int[][] WhiskyData = {
                {1, 0, 1, 1},
                {1, 1, 1, 1},
                {1, 0, 0, 1},
                {1, 0, 0, 1},
                {0, 1, 0, 1},
                {0, 1, 1, 0},
                {0, 1, 1, 0},
                {0, 1, 1, 0},
                {0, 0, 1, 0},
                {0, 0, 1, 0}
        };*/
        int[][] WhiskyData = {
                {1, 1, 1, 1, 0, 0, 0, 0, 0, 0}, // Peaty
                {0, 1, 0, 0, 1, 1, 1, 1, 0, 0}, // Woody
                {1, 1, 0, 0, 0, 1, 1, 1, 1, 1}, // Sweet
                {1, 1, 1, 1, 1, 0, 0, 0, 0, 0}  // Region
        };

        System.out.println("===============================");
/*
        int[][] peatyObsTab = {
                {4, 0}, // High/Yes (1)
                {1, 5} // Low/No (0)
        };
*/
        int[][] peatyObsTab = { // Tony's table, used as a 2nd set of data to verify the infoGain and Chi-Squared calc
                {2, 3}, // Sunny
                {4, 0}, // Overcast
                {3, 2} // Sunny
        };
/*        int[][] peatyObsTab = { // https://www.kdnuggets.com/2020/02/decision-tree-intuition.html ConTab for
                                    // verifying Gini calc. Accessed: 10/05/2022.
                {3, 2}, // Sunny
                {4, 0}, // Overcast
                {3, 2} // Sunny
        };*/

        System.out.println("\nContingency Table (Peaty): \n" + Arrays.deepToString(peatyObsTab).replace("], ", "]\n") + "\n");

        System.out.println("===============================");
        double measuredInfoGain = measureInformationGain(peatyObsTab);
        System.out.println("measure <Information Gain> for Peaty = " + measuredInfoGain);

        System.out.println("===============================");
        double measuredInfoGainRatio = measureInformationGainRatio(peatyObsTab);
        System.out.println("measure <Information Gain Ratio> for Peaty = " + measuredInfoGainRatio);

        System.out.println("===============================");
        double measureGini = measureGini(peatyObsTab);
        System.out.println("measure <Measure Gini> for Peaty = " + measureGini);

        System.out.println("===============================");
        double measureChiSquare = measureChiSquared(peatyObsTab);
        System.out.println("measure <Measure Chi Square> for Peaty = " + measureChiSquare);


/*        System.out.println("===============================");
        int[][] woodyObsTab = {
                {2, 3}, // High/Yes (1)
                {3, 2} // Low/No (0)
        };
        System.out.println("\nContingency Table (Woody): \n" + Arrays.deepToString(woodyObsTab).replace("], ", "]\n") + "\n");

        double measuredInfoGain2 = measureInformationGain(woodyObsTab);
        System.out.println("==> (Woody) Info Gain: " + measuredInfoGain2);

        System.out.println("===============================");
        int[][] sweetObsTab = {
                {2, 5}, // High/Yes (1)
                {3, 0} // Low/No (0)
        };

        System.out.println("\nContingency Table (Sweet): \n" + Arrays.deepToString(sweetObsTab).replace("], ", "]\n") + "\n");

        double measuredInfoGain3 = measureInformationGain(sweetObsTab);
        System.out.println("==> (Sweet) Info Gain: " + measuredInfoGain3);*/
    }
}
