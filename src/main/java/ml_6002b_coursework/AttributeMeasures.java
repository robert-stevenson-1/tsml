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

        //calc the total
        for (int[] attriVal :
                table) {
            //sum the observed values for the current attriVal and add to the total tally
            int attriValTotal = Arrays.stream(attriVal).sum();
            attriValTotals.add(attriValTotal);
            total += attriValTotal;
        }
        System.out.println("Total: " + total);

        //get the totals observation scores and calc entropy for each attribute
        for (int a = 0; a < table.length; a++){ // loop through each different attribute value
            double tempEntropy = 0.0;
            int tempClassTotal = 0;
            //calc the Entropy for each attribute value's class values
            for (int i = 0; i < table[a].length; i++) {
                tempEntropy += (table[a][i]/(double)attriValTotals.get(a))*log2Ent(table[a][i]/(double)attriValTotals.get(a));
                tempClassTotal += table[i][a];
            }
            tempEntropy *= -1;
            entropy.add(tempEntropy);
            obsClassTotals.add(tempClassTotal);
        }
        System.out.println("AttriValTotals: " + attriValTotals);
        System.out.println("ObsScoreTotal: " + obsClassTotals);
        System.out.println("Entropy: " + entropy);

        //get root entropy and add it to the information gain ready to subtract from
        for (int i = 0; i < obsClassTotals.size(); i++) {
            infoGain += (obsClassTotals.get(i)/total)*log2Ent(obsClassTotals.get(i)/total);
        }
        infoGain *= -1;
        System.out.println("Entropy of Root: " + infoGain);

        //calc the information gain
        for (int i = 0; i < entropy.size(); i++) {
            infoGain -= ((attriValTotals.get(i)/total)*entropy.get(i));
        }

        return infoGain;
    }

    public static double measureInformationGainRatio(int[][] table){
        return 0.0;
    }

    public static double measureGini(int[][] table){
        return 0.0;
    }

    public static double measureChiSquared(int[][] table){
        return 0.0;
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
                {1,1,1,1,0,0,0,0,0,0}, // Peaty
                {0,1,0,0,1,1,1,1,0,0}, // Woody
                {1,1,0,0,0,1,1,1,1,1}, // Sweet
                {1,1,1,1,1,0,0,0,0,0}  // Region
        };

        System.out.println("===============================");
        int[][] peatyObsTab = {
                {4, 0}, // High/Yes (1)
                {1, 5} // Low/No (0)
        };

        System.out.println("\nContingency Table (Peaty): \n" + Arrays.deepToString(peatyObsTab).replace("], ", "]\n") + "\n");

        double measuredInfoGain = measureInformationGain(peatyObsTab);
        System.out.println("==> (Peaty) Info Gain: " + measuredInfoGain);

        System.out.println("===============================");
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
        System.out.println("==> (Sweet) Info Gain: " + measuredInfoGain3);
    }

}
