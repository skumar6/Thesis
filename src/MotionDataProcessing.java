import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by Sanjeev on 8/21/2016.
 */
public class MotionDataProcessing {

    public static void main(String[] args) {
        // TODO Auto-generated method stub
        BufferedReader br;
        String sLine;
        String[] sarr = new String[6];
        List<double[]> listData = new ArrayList<double[]>();
        List<double[]> chunkified = new ArrayList<double[]>();
        List<Double> xacc = new ArrayList<Double>();
        List<Double> rmsacc = new ArrayList<Double>();

        String fileName = "o9demost";      //file pulled from server containing inertial sensor data.
        ArrayList<ArrayList<Double>> finalList = new ArrayList<ArrayList<Double>>();

        String peaks4matlab = "C:/Users/Sanjeev/Desktop/plotHelpers/peaks.txt";
        BufferedWriter peaksWriter;


        String trainingSetPath = "C:/Users/Sanjeev/Desktop/arffs/motionArffs/motionTrainingSet.arff";
        String testSetPath= "C:/Users/Sanjeev/Desktop/arffs/motionArffs/"+fileName+".arff";
        String resultDestination = "C:/Users/Sanjeev/Desktop/"+fileName+".arff";
        File dataForClassification = new File(testSetPath);
        BufferedWriter  arffFileWriter;

        try {
            br = new BufferedReader(new FileReader("C:/Users/Sanjeev/Desktop/test&PlotFiles/" + fileName + ".txt"));
            while ((sLine = br.readLine()) != null) {
                sarr = sLine.split(" ");
                double[] darr = new double[sarr.length];
                xacc.add(Double.parseDouble(sarr[3]));
                rmsacc.add(Double.parseDouble(sarr[2]));
                for (int i = 0; i < sarr.length; i++) {
                    darr[i] = Double.parseDouble(sarr[i]);
                }
                listData.add(darr);
            }
            br.close();

            List<Double> smoothenX = movingAverage(xacc, 10);           //smoothen using moving window of size 10
            List<Integer> peaksFinal = peakFinder2(smoothenX);         //on this smoothen data, determine start-end points.
           // List<Integer> peaksFinal= adjustPeaks(peaksFinal1, 10);

            peaksWriter = new BufferedWriter(new FileWriter(peaks4matlab));  //write it somewhere for matlab plotting
            System.out.println("size in final peaks:  " + peaksFinal.size());
            for (int k = 0; k < peaksFinal.size(); k++) {
                System.out.print(peaksFinal.get(k) + " ");
                peaksWriter.write(peaksFinal.get(k)+" ");
            }
            peaksWriter.close();
            System.out.println();



            //chunkified is a list of double[], we have data read from initial file containing all sensor data
            //in dataList, and all peaks in peaksFinal1, read from datalist and store in chukified, data between
            //two peaks, these peaks are pointers for start and end of motion.
            for (int i = 1; i < peaksFinal.size(); i = i + 2) {  //writing chunk only between peak points
                int start = peaksFinal.get(i - 1);
                int end = peaksFinal.get(i);
                while (start++ <= end) {
                    chunkified.add(listData.get(start));
                }
                System.out.println("chunkified size" + chunkified.size());
                finalList.add(dataProcessing(chunkified)); //data processing on chunkified gives us feature set, dataProcessing()
                //uses 7 fields of list of double array and produces 4 values min, max, mean, standard deviation corresponding to
                //each field. and first value in the returned dlist is timestamp.total length returned from dataProcessing is
                //(1+7*4= 29)
                System.out.println("finallist size is :" + finalList.size());
                chunkified = new ArrayList<double[]>();
            }



            //this finallsit of 29 values is our feature set, write the header of .arff file and append this list
            //it ll become a proper .arff file.
            arffFileWriter = new BufferedWriter(new FileWriter(dataForClassification));
            arffFileWriter.write(getArffHeader(fileName));
            for(int i =0; i<finalList.size(); i++){
                for(int k =0; k<finalList.get(i).size(); k++){
                    arffFileWriter.write(finalList.get(i).get(k)+",");
                }
                arffFileWriter.write("?");
                arffFileWriter.write("\n");
            }
            arffFileWriter.flush();
            arffFileWriter.close();

            System.out.println("**--10 fold cross validation results--**");
            crossValidate10fold(trainingSetPath);  //cross validate and see the results.



            System.out.println("***********--Classification Results--***********");
            classification(trainingSetPath, testSetPath, resultDestination );  //classification and results.

            //no current use of movingVariance(), used initially for plotting and analysis.
            List<Double> movingVar = movingVariance(smoothenX, 10);
            File file = new File("C:/Users/Sanjeev/Desktop/file4plot/" + fileName + "4plots.txt");
            BufferedWriter bw = new BufferedWriter(new FileWriter(file));
            for (int l = 0; l < movingVar.size(); l++) {
                bw.write(l + " " + rmsacc.get(l) + " " + xacc.get(l) + " " + smoothenX.get(l) + " " + movingVar.get(l) + "\n");
            }
            bw.close();

            //List<Integer> peaksInRawStd = peakFinder(rmsacc, 1.2);



        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }









        //////////////////////////////////////////////////
    ////helper methods //////////////////////////////

    //pass a list, return peaks in that list,
    //based on vicinity and time window filtering,peaks here are start-end points
    private static List<Integer> peakFinder(List<Double> arr, double threshold){
        List<Integer> result = new ArrayList<Integer>();
        int j =-5;
        for(int i =1; i<arr.size()-1; i++){
            boolean peak = isPeak(arr.get(i-1), arr.get(i), arr.get(i+1),threshold);
            if(peak){
                if((i-j)>5){
                    j=i;
                    result.add(i);
                }
            }
        }
        return result;
    }

    //see if given point is a peak
    private static boolean isPeak(double a, double b, double c, double threshold){
        if( (b>a)&& (b>c) && b>=threshold ) return true;
        else return false;
    }


    //method for start and end point detection
    //based on thresholds.
    private static List<Integer> peakFinder2(List<Double> arr){
        double lth= 0.5;
        double hth =10;
        List<Integer> result = new ArrayList<Integer>();
        for(int i =6; i<arr.size()-5; i++){
            double val = arr.get(i);
            if(lth == 0.5 && val< lth){
                result.add(i);
                hth = .7;
                lth =-10;
            } if(hth ==.7 && val>hth){
                result.add(i+5);
                lth = 0.5;
                hth = 10;
            }
        }
        return result;
    }

    //calculating moving average over a window to smooth sensor data
    private static List<Double> movingAverage(List<Double> data, int window){
        List<Double> result = new ArrayList<Double>();
        for(int i =0; i<data.size()-window; i++){
            double sum =0;
            for(int j = i; j<i+window; j++){
                sum = sum + data.get(j);
            }
            double avg = sum/window;
            result.add(avg);
        }
        return result;
    }


    //getting moving variance for a selected window to see fluctuations in data
    private static List<Double> movingVariance (List<Double> data, int window){
        List<Double> result = new ArrayList<Double>();
        for(int i =0; i<data.size()-window; i++){
            double sum =0;
            double var =0;
            for(int j =i; j<i+window; j++){
                sum = sum+data.get(j);
            }
            double mean = sum/window;
            for(int j =i; j<i+window; j++){
                var = var + (mean-data.get(j))*(mean-data.get(j));
            }
            var = var/window;
            result.add(var);
        }
        return result;
    }

    //helper to get std devaition, min max,mean on chunkified data.
    private static ArrayList<Double> dataProcessing(List<double []> chunkified){
        int chunk =chunkified.size();
        ArrayList<Double> dlist = new ArrayList<Double>();

        dlist.add(chunkified.get(0)[1]);

        for(int i =2; i<9; i++){
            double min = Double.MAX_VALUE;
            double max = Double.MIN_VALUE;
            double sum = 0;
            double mean = 0;

            for(int j =0; j<chunk; j++){
                double val = (chunkified.get(j)[i]);
                if(val >max ) max = val;
                if(val < min) min = val;
                sum = sum+ val;
            }
            //use sum taken from previous iteration, calculate the mean.
            mean = sum/chunk;
            double temp =0;

            //calculated the mean, now calculating standard deviation.
            for(int j =0; j<chunk; j++){
                double diff = mean-(chunkified.get(j)[i]);
                temp = temp + (diff*diff);
            }
            double std = Math.sqrt((temp/chunk));
            //push all these values to the temporary list
            dlist.add(min);
            dlist.add(max);
            dlist.add(mean);
            dlist.add(std);
        }
        //add this temporary list to final list.

        return dlist;
    }

    //generate arff header for weka classification.
    private static String getArffHeader(String textFileName){
        String header = "@relation " +textFileName+ "\n \n"
                + "@attribute timestamp numeric \n"

                + "@attribute rmsAmin numeric \n"
                + "@attribute rmsAmax numeric \n"
                + "@attribute rmsAmean numeric \n"
                + "@attribute rmsAstd numeric \n"

                + "@attribute xAmin numeric \n"
                + "@attribute xAmax numeric \n"
                + "@attribute xAmean numeric \n"
                + "@attribute xAstd numeric \n"

                + "@attribute yAmin numeric \n"
                + "@attribute yAmax numeric \n"
                + "@attribute yAmean numeric \n"
                + "@attribute yAstd numeric \n"

                + "@attribute zAmin numeric \n"
                + "@attribute zAmax numeric \n"
                + "@attribute zAmean numeric \n"
                + "@attribute zAstd numeric \n"

                + "@attribute xGmin numeric \n"
                + "@attribute xGmax numeric \n"
                + "@attribute xGmean numeric \n"
                + "@attribute xGstd numeric \n"

                + "@attribute yGmin numeric \n"
                + "@attribute yGmax numeric \n"
                + "@attribute yGmean numeric \n"
                + "@attribute yGstd numeric \n"

                + "@attribute zGmin numeric \n"
                + "@attribute zGmax numeric \n"
                + "@attribute zGmean numeric \n"
                + "@attribute zGstd numeric \n"

                + "@attribute level {yes,no} \n \n"
                + "@data \n";
        return header;
    }


    //finally, perform classification
    private static void classification(String trainingSetPath, String testSetPath, String DestinationPath){
        BufferedReader br = null;
        BufferedWriter timeWriter = null;
        String time4matlab = "C:/Users/Sanjeev/Desktop/plotHelpers/classifyYesTime.txt";
        try {
            br = new BufferedReader
                    (new FileReader(trainingSetPath));
            Instances train = new Instances(br);
            train.setClassIndex(train.numAttributes()-1);

            br = new BufferedReader
                    (new FileReader(testSetPath));
            Instances test = new Instances(br);
            test.setClassIndex(train.numAttributes()-1);

            br.close();

            NaiveBayes j48 = new NaiveBayes();
            j48.buildClassifier(train);

            Instances lebeled = new Instances(test);

            timeWriter = new BufferedWriter(new FileWriter(time4matlab));
            for(int i =0; i<test.numInstances(); i++){
                double clebel = j48.classifyInstance(test.instance(i));
                if(clebel == 0)
                    timeWriter.write(test.instance(i).value(0)+" ");

                java.util.Date time=new java.util.Date((long)test.instance(i).value(0));
                System.out.println(clebel +" : "+ time);
                lebeled.instance(i).setClassValue(clebel);
            }
            timeWriter.close();

            BufferedWriter writer = new BufferedWriter(new FileWriter(DestinationPath));
            writer.write(lebeled.toString());
            writer.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    public static void crossValidate10fold(String trainingSetPath){
        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(trainingSetPath));
            Instances train = new Instances(br);
            train.setClassIndex(train.numAttributes()-1);
            br.close();

            NaiveBayes nb = new NaiveBayes();
            nb.buildClassifier(train);
            Evaluation eval = new Evaluation(train);
            eval.crossValidateModel(nb, train, 10, new Random(1));
            System.out.println(eval.toSummaryString("---showing results for naiveBayes---", true));
            System.out.println(eval.fMeasure(1)+" "+ eval.recall(1)+" "+eval.precision(1));
            System.out.println(eval.toMatrixString());

            J48 j48 = new J48();
            j48.buildClassifier(train);
            Evaluation eval2 = new Evaluation(train);
            eval2.crossValidateModel(j48, train, 10, new Random(1));
            System.out.println(eval2.toSummaryString("---showing results for j48---", true));
            System.out.println(eval2.fMeasure(1)+" "+ eval2.recall(1)+" "+eval2.precision(1));
            System.out.println(eval2.toMatrixString());

            RandomForest rForest = new RandomForest();
            rForest.buildClassifier(train);
            Evaluation eval3 = new Evaluation(train);
            eval3.crossValidateModel(rForest, train, 10, new Random(1));
            System.out.println(eval3.toSummaryString("---showing results for RandomForest---", true));
            System.out.println(eval3.fMeasure(1)+" "+ eval3.recall(1)+" "+eval3.precision(1));
            System.out.println(eval3.toMatrixString());
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // getting eucledian distance list for a given window
    private static List distanceWindow(List<Double> data, int windowLen){
        List result = new ArrayList<Double>();
        double[] window = new double[windowLen];
        for(int i =0; i<windowLen; i++ ){
            window[i]= data.get(i);
        }
        for(int i = windowLen; i<data.size()-windowLen; i++){
            double euclideanDistance = 0;
            int index =0;
            for(int j =i; j<i+windowLen; j++){
                euclideanDistance = euclideanDistance + (window[index]-data.get(j))*(window[index]-data.get(j));
            }
            result.add(euclideanDistance);
        }
        System.out.println(result.size()+	" result size...");;
        return result;
    }

    private static void adjustPeaks(List<Integer> peaksFinal, int window){
        for(int i1 =0; i1<peaksFinal.size(); i1++){
            if(i1%2==0)
                peaksFinal.add(peaksFinal.get(i1));
            else
                peaksFinal.add(peaksFinal.get(i1)+(window+2));
        }
    }

    private static ArrayList<ArrayList> movingWindowdata (List<double []> data){
        int length = data.size();
        int c =0;
        ArrayList<ArrayList> instances= new ArrayList<ArrayList>();
        while(c+40<length){
            ArrayList<Double> dlist = new ArrayList<Double>();
            for(int i =2; i<9; i++){
                double min = Double.MAX_VALUE;
                double max = Double.MIN_VALUE;
                double sum = 0;
                double mean = 0;
                int j =0;
                while(j++<40){
                    double val = (data.get(c+j)[i]);
                    if(val >max ) max = val;
                    if(val < min) min = val;
                    sum = sum+ val;
                }
                mean = sum/40;
                double temp =0;
                for(int k =0; j<40; j++){
                    double diff = mean-(data.get(c+j)[i]);
                    temp = temp + (diff*diff);
                }
                double std = Math.sqrt((temp/40));
                dlist.add(min);
                dlist.add(max);
                dlist.add(mean);
                dlist.add(std);
            }
            instances.add(dlist);
            c=c+20;
        }
        return instances;
    }

    //designed low pass filter//never used
    private static double[] lpf (List<Double> rmsacc){
        double alpha = .25;
        double[] output = new double[rmsacc.size()];
        output[0]= (double) rmsacc.get(0);
        if(output == null) return output;

        for(int i =1; i<rmsacc.size(); i++){
            output[i]= output[i-1] + alpha*(rmsacc.get(i)- output[i-1]);
        }

        return output;
    }

    private static List finalPeaks(List<double[]> listData){
        File file = new File("C:/Users/Sanjeev/Desktop/file.txt");
        List<Double> rmsPeakList = new ArrayList<Double>();

        int window = 10;
        int indexx = 3;
        int indexr = 2;
        List<Double> varxList = new ArrayList<Double>();
        List<Double> varrList = new ArrayList<Double>();
        List<Integer> finalPeaks = new ArrayList<Integer>();
        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(file));


            for(int i =0; i<listData.size()-window; i++){
                rmsPeakList.add(listData.get(i)[2]);
                double sumx = 0;
                double sumr = 0;
                double varx =0;
                double varr =0;
                for(int j =i; j<i+window; j++){
                    sumx = sumx + listData.get(j)[indexx];
                    sumr = sumr + listData.get(j)[indexr];
                }

                double meanx = sumx/window;
                double meanr = sumr/window;

                for(int j =i; j<i+window; j++){
                    varx  = (meanx-listData.get(j)[indexx])*(meanx-listData.get(j)[indexx]);
                    varr  = (meanr-listData.get(j)[indexr])*(meanr-listData.get(j)[indexr]);
                }
                varx = varx/window;
                varr = varr/window;
                String s = varx +"  "+varr +"\n";
                bw.write(s);
                // bw.close();

                varxList.add(varx);
                varrList.add(varr);
            }
            List<Integer> peaksInX= peakFinder(varxList, (5.00E-02));
            List<Integer> peaksInrms = peakFinder(rmsPeakList, 1.2);
            for(int m =0; m<peaksInrms.size(); m++ ){
                System.out.print(peaksInrms.get(m)+" ");
            }
            System.out.println(" ");
            for(int i1 =0; i1<peaksInX.size(); i1++){
                if(i1%2==0)
                    finalPeaks.add(peaksInX.get(i1)+window/2);
                else
                    finalPeaks.add(peaksInX.get(i1)+(window));
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
        return finalPeaks;
    }





}
