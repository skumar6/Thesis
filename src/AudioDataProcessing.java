import org.jtransforms.fft.DoubleFFT_1D;
import weka.classifiers.trees.RandomTree;
import weka.core.Instances;

import javax.sound.sampled.*;
import java.io.*;
import java.nio.ByteBuffer;
import java.util.ArrayList;

/**
 * Created by Sanjeev on 8/12/2016.
 */
public class AudioDataProcessing {

    static String timeStamp  = "1473224499204";

    public static void main(String[] args){

        // in this program we are using a number of IOs, so that we can plot and compare are results at each point
        //with original signal in matlab.
        String filename= "1473223559708";
        ArrayList<Double> hilbertFil = new ArrayList<Double>();
        String path = "C:/Users/Sanjeev/Desktop/soundClips/"+filename+".wav";  //where original audio file is stored.
        String byteToDoublePath = "C:/Users/Sanjeev/Desktop/audioJava/sampleText.txt";  //store converted double[] file from audio byte[] file
        String movingAvgPath = "C:/Users/Sanjeev/Desktop/audioJava/MovingAvg.txt";
        String pointersPath = "C:/Users/Sanjeev/Desktop/audioJava/pointers.txt"; //start-end points
        String hilbertOutPath = "C:/Users/Sanjeev/Desktop/audioJava/hilbertOutput.txt"; //data after hilbert transform.
        String featureSetPath = "C:/Users/Sanjeev/Desktop/arffs/audioArffs/audioTestSet.arff"; //feature set
        String timeSeriesChunksPath = "C:/Users/Sanjeev/Desktop/audioJava/timeSeriesChunks.txt"; //extracted time series chunk path.
        String fftChunksPath = "C:/Users/Sanjeev/Desktop/audioJava/fftChunks.txt"; //chunk after fft

        String trainingSetPath = "C:/Users/Sanjeev/Desktop/arffs/audioArffs/audioTrainingSet.arff";
        String testSetPath = "C:/Users/Sanjeev/Desktop/arffs/audioArffs/audioTestSet.arff";
        String DestinationPath = "C:/Users/Sanjeev/Desktop/results.arff"; //result

        int duration = (int) (getAudioLength(path)*1000);

        ArrayList<ArrayList<Double>> timeSeriesChunks = new ArrayList<ArrayList<Double>>();
        ArrayList<double[]> fftChunks = new ArrayList<double[]>();

        ArrayList<Double> finals= audioRead(path, hilbertFil);
        writeFileDoublelist(hilbertFil, hilbertOutPath);
        writeFileDouble(finals, byteToDoublePath );

        //moving average has smoothen values, used for peak detection
        double[] MovingAvg = movingAvg(hilbertFil,2500);
        writeFile(MovingAvg, movingAvgPath);

        //pointers are the point, in between which, we detects voice signals we
        //looking for.
        ArrayList<Integer> pointers ;
        pointers = chunkInPair(MovingAvg);
        writeFileInteger(pointers, pointersPath);
        System.out.println("Number of peaks: "+pointers.size());

        //start chunkification
        ArrayList<double[]> timeSeriesFeatureSet = new ArrayList<double[]>();
        ArrayList<double[]> fftFeatureSet = new ArrayList<double[]>();
        double[] chunkArray;
        int i =0;

        while(i < pointers.size() ){
            double[] timeSeriesFeatures = new double[5];
            int start = pointers.get(i);
            int end = pointers.get(i+1);
            ArrayList<Double> chunk = chunkify(finals, start, end);

            timeSeriesChunks.add(chunk);

            timeSeriesFeatures[0] = getTimeStamp(start, duration);
            double[] temp = getFeaturea(chunk);
            for(int k =0; k <temp.length; k++){
                timeSeriesFeatures[k+1]= temp[k];
            }
            timeSeriesFeatureSet.add(timeSeriesFeatures);


            //preparing for fft and frequency feature extraction
            chunkArray = new double[chunk.size()];
            for (int k = 0; k < chunkArray.length; k++) {
                chunkArray[k] = chunk.get(k);                // java 1.5+ style (outboxing)
            }
            double[] fftchunk = fft(chunkArray);

            fftChunks.add(fftchunk);
            double[] fftFeatures = getFeature(fftchunk);
            fftFeatureSet.add(fftFeatures);


            i=i+2;
        }
        System.out.println("no of fft chunks: "+fftChunks.size());
        writeMatrixList(timeSeriesChunks, timeSeriesChunksPath);
        writeMatrixArr(fftChunks, fftChunksPath);

        File featureSetfile = new File(featureSetPath);
        try {
            BufferedWriter featureSetWriter = new BufferedWriter(new FileWriter(featureSetfile));
            featureSetWriter.write(getArffHeader(filename));
            for(int j=0; j<timeSeriesFeatureSet.size(); j++) {
                featureSetWriter.write(timeSeriesFeatureSet.get(j)[0]+","+timeSeriesFeatureSet.get(j)[1]+","+timeSeriesFeatureSet.get(j)[2]+","
                                            +timeSeriesFeatureSet.get(j)[3]+"," +timeSeriesFeatureSet.get(j)[4]+","
                                            +fftFeatureSet.get(j)[0]+ ","+ fftFeatureSet.get(j)[1]+","
                                            +fftFeatureSet.get(j)[2]+ ","+ fftFeatureSet.get(j)[3]+  ",?"+ "\n");
            }
            featureSetWriter.close();

            classification(trainingSetPath, testSetPath, DestinationPath );
        } catch (IOException e) {
            e.printStackTrace();
        }

    }







    //takes start and end index which are pair of peaks in this case and then chunkify the initial set over it.
    public static ArrayList<Double> chunkify(ArrayList<Double> finals, int start, int end){
        ArrayList<Double> chunk = new ArrayList<Double>();
        for(int i= start; i<=end; i++) {
            chunk.add(finals.get(i));
        }
        return chunk;
    }

    //now when we have the chunk process it to get feature set which ll be min, max mean and standard devaition in
    // this case.
    public static double[] getFeaturea(ArrayList<Double> arr) {
        double[] featureSet = new double[4];
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;
        double sum = 0;
        double mean = 0;
        for (int i = 0; i < arr.size(); i++) {
            double val = arr.get(i);
            if (val > max)
                max = val;
            if (val < min)
                min = val;
            sum = sum + val;
        }
        mean = sum / arr.size();
        double varsum = 0;
        for (int i = 0; i < arr.size(); i++) {
            varsum = varsum + ((mean - arr.get(i)) * (mean - arr.get(i)));
        }
        varsum = varsum / arr.size();

        featureSet[0] = min;
        featureSet[1] = max;
        featureSet[2] = mean;
        featureSet[3] = varsum;

        return featureSet;
    }

    public static double[] getFeature(double[] arr){
        double[] featureSet = new double[4];
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;
        double sum = 0;
        double mean = 0;
        for (int i = 0; i < arr.length; i++) {
            double val = arr[i];
            if (val > max)
                max = val;
            if (val < min)
                min = val;
            sum = sum + val;
        }
        mean = sum / arr.length;
        double varsum = 0;
        for (int i = 0; i < arr.length; i++) {
            varsum = varsum + ((mean - arr[i]) * (mean - arr[i]));
        }
        varsum = varsum / arr.length;

        featureSet[0] = min;
        featureSet[1] = max;
        featureSet[2] = mean;
        featureSet[3] = varsum;

        return featureSet;
    }

    public static double[] movingAvg(ArrayList<Double> arr, int windowSize){
        int len = arr.size();
        int j =0;
        double[] movingAvgArr = new double[len];
        for(int i = 0; i<len; i++){
            int k =i;
            double sum =0;
            if(k+windowSize<=len){
                for(k=i; k<i+windowSize; k++){
                    sum = sum+arr.get(k);
                }
                movingAvgArr[j++]=sum/windowSize;
            }
        }
        return movingAvgArr;
    }

    //extract start and end pointer.
    public static ArrayList<Integer> chunkInPair(double[] arr){
        double hth= 0.1;
        double lth = 0.1;
        ArrayList<Integer> pointers = new ArrayList<Integer>();

        for(int i =0; i<arr.length; i++) {
            if (hth == 0.1 && arr[i] > hth) {
                pointers.add(i - 2500);
                hth = 2;
                lth = 0.05;
            }

            if (lth == 0.05 && arr[i] < lth) {
                pointers.add(i);
                hth = .1;
                lth = -1;
            }
        }
        return pointers;
    }

    //normal file writing
    public static void writeFile(double[] arr, String path){
        String samplefile = path;
        BufferedWriter bw;
        try {
            bw = new BufferedWriter(new FileWriter((samplefile)));
            for(int i=0; i<arr.length; i++){
                bw.write(arr[i] + " ");
            }
            bw.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    //perform FFT
    public static double[] fft(double[] input){

        DoubleFFT_1D fftDo = new DoubleFFT_1D(input.length);
        double[] fft = new double[input.length * 2];
        System.arraycopy(input, 0, fft, 0, input.length);
        fftDo.realForwardFull(fft);
        double[] frequency = new double[fft.length/2];

        for(int i =0; i<input.length/2; i++){
            double real = fft[2*i];
            double img = fft[2*i +1];
            frequency[i] = Math.sqrt(real*real + img*img);
        }
        String fftPath = "C:/Users/Sanjeev/Desktop/audioJava/fft.txt";
        writeFile(frequency, fftPath);
//        for(double d: frequency)
//            System.out.println("f: " + d);
        //
        return frequency;
    }


    //given access to the audiofile, read that file, it is in form of byte[] first some of the byte elements are header
    //information depending upon type of encoding used, here first 44 bytes are header. Once byte array, containing only
    // audio data is fetched we need to collect and unify samples to form the exact the signal. byte[] to double[].
    //along with that we are also doing hilbert filtering and saving it in another list for future purpose.
    //list from hilbert filter is used to determine sudden peaks and drops in audio signal, thoese point are used to
    //break the while audio signal in chunks.
    //fft will be performed on these chunks to derive featires from audio frequency domain.

    public static ArrayList<Double> audioRead(String audioFilePath, ArrayList<Double> hilbertFil) {
        int totalFramesRead = 0;
        int i = 0;
        int k = 0;
        int numBytesRead = 0;
        int numFramesRead = 0;
        String path = audioFilePath;
        File fileIn = new File(path);
        AudioInputStream audioInputStream = null;
        ArrayList<Double> finals = new ArrayList<Double>();

        try {
            audioInputStream = AudioSystem.getAudioInputStream(fileIn);
            long numberOfFrames = audioInputStream.getFrameLength();
            double rate = AudioSystem.getAudioFileFormat(fileIn).getFormat().getFrameRate();
            System.out.println("Audio duration:  "+numberOfFrames/rate );
            int bytesPerFrame = audioInputStream.getFormat().getFrameSize();

            int numBytes = (int) (numberOfFrames * bytesPerFrame);

            System.out.println("format info: " + numberOfFrames);
            System.out.println(AudioSystem.getAudioFileFormat(fileIn));
            byte[] audioBytes = new byte[numBytes];

            while ((numBytesRead = audioInputStream.read(audioBytes)) != -1) {
                numFramesRead = numBytesRead / bytesPerFrame;
                totalFramesRead += numFramesRead;
            }
            //finals is final double values collected after wavread.
            while (i < audioBytes.length) {
                int val = ((audioBytes[i] & 0xff) |
                        ((audioBytes[i + 1] & 0xff) << 8) |
                        ((audioBytes[i + 2] & 0xff) << 16) |
                        ((audioBytes[i + 3]) << 24));
                //for doubles they take values between [0,1] so we need to devide it by 2^31-1
                finals.add(val / (Math.pow(2, 31) - 1));
                if( (val/( Math.pow(2, 31)-1) >=0))
                    hilbertFil.add( val/( Math.pow(2, 31)-1));
                else
                    hilbertFil.add(0.0);
                i= i+4;
            }
        } catch (UnsupportedAudioFileException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return finals;
    }

    public static void writeFileDouble(ArrayList<Double> arr, String path){
        String samplefile = path;
        BufferedWriter bw;
        try {
            bw = new BufferedWriter(new FileWriter((samplefile)));
            for(int i=0; i<arr.size(); i++){
                bw.write(arr.get(i) + " ");
            }
            bw.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void writeFileInteger(ArrayList<Integer> arr, String path){
        String samplefile = path;
        BufferedWriter bw;
        try {
            bw = new BufferedWriter(new FileWriter((samplefile)));
            for(int i=0; i<arr.size(); i++){
                bw.write(arr.get(i) + " ");
            }
            bw.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void writeFileDoublelist(ArrayList<Double> arr, String path){
        String samplefile = path;
        BufferedWriter bw;
        try {
            bw = new BufferedWriter(new FileWriter((samplefile)));
            for(int i=0; i<arr.size(); i++){
                bw.write(arr.get(i) + " ");
            }
            bw.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void writeMatrixArr(ArrayList<double[]> matrix, String matrixPath){
        String path = matrixPath;
        File file = new File(path);
        BufferedWriter matrixWriter = null;
        try {
            matrixWriter = new BufferedWriter(new FileWriter(file));
            int numArrays = matrix.size();
            for(int i=0; i< numArrays; i++){
                double[] arr = matrix.get(i);
                for(int j=0; j< arr.length; j++){
                    double val = arr[j];
                    matrixWriter.write(val+"\t");
                }
                matrixWriter.write("\n");

            }
            matrixWriter.close();


        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    public static void writeMatrixList (ArrayList<ArrayList<Double>> matrix, String matrixPath){
        String path = matrixPath;
        File file = new File(path);
        BufferedWriter matrixWriter = null;
        try {
            matrixWriter = new BufferedWriter(new FileWriter(file));
            int numArrays = matrix.size();
            for(int i=0; i< numArrays; i++){
                ArrayList<Double> arr = matrix.get(i);
                for(int j=0; j< arr.size(); j++){
                    double val = arr.get(j);
                    matrixWriter.write(val+" ");
                }
                matrixWriter.write("\n");
            }
            matrixWriter.close();


        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    public static long getTimeStamp(int frameIndex, int duration){
        long timestamp = Long.parseLong(timeStamp)-duration+10000;
        double timePerFrame = 1000.0/44100;
        return (long) (timestamp+ (frameIndex*timePerFrame));
    }

    public static double getAudioLength(String audioFilePath){
        String path = audioFilePath;
        File fileIn = new File(path);
        AudioInputStream audioInputStream = null;
        double duration=0;
        try {
            audioInputStream = AudioSystem.getAudioInputStream(fileIn);
            long numberOfFrames = audioInputStream.getFrameLength();
            double rate = AudioSystem.getAudioFileFormat(fileIn).getFormat().getFrameRate();
            duration = numberOfFrames/rate;
            System.out.println("Audio duration:  "+numberOfFrames/rate );
        } catch (UnsupportedAudioFileException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return duration;
    }

    private static void classification(String trainingSetPath, String testSetPath, String DestinationPath){
        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(trainingSetPath));
            Instances train = new Instances(br);
            train.setClassIndex(train.numAttributes()-1);

            br = new BufferedReader(new FileReader(testSetPath));
            Instances test = new Instances(br);
            test.setClassIndex(train.numAttributes()-1);

            br.close();
//            Remove rm = new Remove();
//            rm.setAttributeIndices("1");  // remove 1st attribute


            //NaiveBayes j48 = new NaiveBayes();
            //J48 j48 = new J48();
            //RandomForest j48 = new RandomForest();
            RandomTree j48 = new RandomTree();
//            FilteredClassifier fc = new FilteredClassifier();
//            fc.setFilter(rm);
//            fc.setClassifier(j48);
            j48.buildClassifier(train);

            Instances lebeled = new Instances(test);
            for(int i =0; i<test.numInstances(); i++){
                double clebel = j48.classifyInstance(test.instance(i));
                System.out.println(test.instance(i).value(0));
                java.util.Date time=new java.util.Date((long)test.instance(i).value(0));
                System.out.println(clebel+" : "+time);
                lebeled.instance(i).setClassValue(clebel);
            }

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

    private static String getArffHeader(String textFileName){
        String header = "@relation " +textFileName+ "\n \n"
                + "@attribute timestamp numeric \n"

                + "@attribute amin numeric \n"
                + "@attribute amax numeric \n"
                + "@attribute amean numeric \n"
                + "@attribute avar numeric \n"

                + "@attribute fmin numeric \n"
                + "@attribute fmax numeric \n"
                + "@attribute fmean numeric \n"
                + "@attribute fvar numeric \n"

                + "@attribute level {yes,no} \n \n"
                + "@data \n";
        return header;
    }

}
