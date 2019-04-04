import jdk.jfr.events.FileReadEvent;

import java.io.*;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Scanner;

/**
 * Created by admin on 2018/2/10.
 */
public class Network {

    public static final double INITIALWEIGHT = 0.5, INITIALBIAS = 0;
    public static String DIRNAME = "C:/Users/admin/Documents/PolyU/Artificial Intelligence/Deep Learning/data";

    // number of layers, and respective number of neurons
    private int layerCount;
    private int[] nodeCount;

    // matrices and list of matrices
    private DynamicMatrix bias;
    private ArrayList<DynamicMatrix> weights;


    // ------ constructors ------



    // input:
    //      weight: ({...}{...}...{...})
    //      bias:   ({...})
    // constructs network

    public Network(File file) {

        // initialize some value
        layerCount = 0;
        ArrayList temp = new ArrayList<Integer>();

        weights = new ArrayList<DynamicMatrix>();
        bias = new DynamicMatrix();

        // read the files and add the values to this object
        try {
            FileReader fr = new FileReader(file);

            // an integer storing the current character
            int tempChar;
            tempChar = fr.read();

            // the weight starts with a '('
            if ((char)tempChar != '(') {
                throw new IllegalArgumentException("Syntax incorrect: Weight should start with a '(', but is a " + (char)tempChar + " instead. \n");
            }

            // creates a string buffer
            StringBuffer sb;

            // the first loop gets all the weights
            while ((char)tempChar != ')') {

                if (tempChar == ' ' || tempChar == '\n' || tempChar == '\t') {
                    tempChar = fr.read();
                    continue;
                }


                // reinitialize the sb in each loop
                sb = new StringBuffer();
                tempChar = fr.read();

                // read until the right bracket
                while ((char)tempChar != '}') {
                    sb.append((char)tempChar);
                    tempChar = fr.read();
                }
                sb.append('}');

                // add the value to this object
                weights.add(new DynamicMatrix(sb.toString()));

                // add the row counts and node counts
                if (weights.size() == 1) {
                    layerCount += 2;
                    temp.add(weights.get(0).sizeColumn(0));
                    temp.add(weights.get(0).sizeRow());
                }
                else {
                    layerCount++;
                    if (weights.get(layerCount-2).sizeColumn(0) != (int)temp.get(layerCount-2)) {
                        throw new IllegalArgumentException("The size of node count dismatch," +
                                " weight " + (layerCount-3) + ": (" + weights.get(layerCount-3).sizeRow() + ", " + weights.get(layerCount-3).sizeColumn(0) + ")," +
                                " weight " + (layerCount-2) + ": (" + weights.get(layerCount-2).sizeRow() + ", " + weights.get(layerCount-2).sizeColumn(0) + ").");
                    }
                    temp.add(weights.get(layerCount-2).sizeRow());
                }
                tempChar = fr.read();

            }


            // once done reading the weights, start reading the bias starting with a '('
            tempChar = fr.read();
            while ((char)tempChar != '(') {
                tempChar = fr.read();
            }
            // the weight starts with a '('
            if ((char)tempChar != '(') {
                throw new IllegalArgumentException("Syntax incorrect: Bias should start with a '('.\n");
            }

            // reinitialize the sb in each loop
            sb = new StringBuffer();
            tempChar = fr.read();
            // the second loop to get the bias

            while ((char)tempChar != ')') {
                // read until the right bracket
                sb.append((char)tempChar);
                tempChar = fr.read();
            }
            // add the value to this object
            bias = new DynamicMatrix((sb.toString()));
            nodeCount = new int[layerCount];
            for (int i = 0; i < layerCount; i++) {
                nodeCount[i] = (int)temp.get(i);
            }

        }
        catch (IOException e) {
            throw new IllegalArgumentException("File not found.\n");
        }
    }

    // construct a network with the number of layers and nodes in separate layers, initializing weights and biases as the final default value.
    public Network(int layerCount, int[] nodeCount) {

        this.layerCount = layerCount;
        this.nodeCount = nodeCount;

        int[] temp = new int[nodeCount.length - 1];
        for (int i = 0; i < temp.length; i++) {
            temp[i] = nodeCount[i+1];
        }
        bias = new DynamicMatrix(layerCount - 1, temp, INITIALBIAS);

        weights = new ArrayList<>();
        for (int i = 0; i < layerCount - 1; i++) {
            weights.add(new DynamicMatrix(nodeCount[i+1], nodeCount[i], INITIALWEIGHT));
        }
    }


    // all layers will be initialized with nodeCount-number of nodes
    public Network(int layerCount, int nodeCount) {

        this.layerCount = layerCount;
        this.nodeCount = DynamicMatrix.getIntArr(layerCount, nodeCount);

        bias = new DynamicMatrix(layerCount - 1, nodeCount, INITIALBIAS);

        weights = new ArrayList<>();
        for (int i = 0; i < layerCount - 1; i++) {
            weights.add(new DynamicMatrix(this.nodeCount[i+1], this.nodeCount[i], INITIALWEIGHT));
        }
    }

    public Network(int layerCount, double[][] bias, double[][][] weights) {
        this.layerCount = layerCount;
        int[] arrBias = new int[bias.length];
        for (int i = 0; i < bias.length; i++) {
            arrBias[i] =bias[i].length;
        }

        this.nodeCount = arrBias;
        this.bias = new DynamicMatrix(bias);

        this.weights = new ArrayList<>();
        for (int i = 0; i < layerCount - 1; i++) {
            this.weights.add(new DynamicMatrix(weights[i]));
        }
    }

    public Network(double[][] bias, double[][][] weights) {
        this(bias.length, bias, weights);
    }


    // ------ learning part ------

    // a forward on one layer, calling the next method
    public DynamicMatrix feedForward(DynamicMatrix stimuli, int layer) {
        if (stimuli.sizeRow() >= 2) {
            throw new IllegalArgumentException("The stimuli should be 1*n.\n");
        }

        // similar as above, while recording the result and return
        DynamicMatrix res = stimuli.transpose();
        res = getWeight(layer).multiplyMtrx(res);
        res.plus(getBias(layer).transpose());
        res = res.transpose();
        res = DynamicMatrix.sigmoid(res);

        return res;
    }

    // ------ a forward on one layer ------
    // If prefer mapping the result into 0-1, use the feedForwardRaw() function.
    // the feedFoward() function uses sigmoid function to make the result between 0-1, and returning A as the 0-1 result,
    // Z as the original calculated result.

    // a feedforward function that calculates the weighted sum and re-evaluate using sigmoid function
    public DynamicMatrix[] feedForward(DynamicMatrix stimuli, int layer, boolean getZ) {
        if (stimuli.sizeRow() != 1) {
            throw new IllegalArgumentException("The stimuli should be 1*n: " + stimuli + " instead.\n");
        }
        if (stimuli.sizeColumn(0) != getNodeCount(0)) {
            throw new IllegalArgumentException("The stimuli is not of nodes' number, node: " + getNodeCount(layer) + ", stimuli: " + stimuli.sizeColumn(0));
        }
        // similar as above, while recording the result and return
        DynamicMatrix layerOutput = stimuli.transpose();
        layerOutput = getWeight(layer).multiplyMtrx(layerOutput);
        layerOutput = layerOutput.transpose();
        layerOutput = layerOutput.plus(getBias(layer));

        // here we already have the weighted and added value of this layer. if the parameter determines to get Z, then we copy the original intermediate value.
        // else, we just return something empty at the position.
        DynamicMatrix ZVal;
        ZVal = layerOutput;

        // this is the step that processes the output with sigmoid to make sure all values fall between 0 and 1
        layerOutput = DynamicMatrix.sigmoid(layerOutput);

        return new DynamicMatrix[] {layerOutput, ZVal};
    }

    // a feedforward function that calculates the weighted sum and reevaluate without using sigmoid function
    //
    public DynamicMatrix[] feedForwardRaw(DynamicMatrix stimuli, int layer, boolean getZ) {
        if (stimuli.sizeRow() != 1) {
            throw new IllegalArgumentException("The stimuli should be 1*n: " + stimuli + " instead.\n");
        }
        if (stimuli.sizeColumn(0) != getNodeCount(0)) {
            throw new IllegalArgumentException("The stimuli is not of nodes' number, node: " + getNodeCount(layer) + ", stimuli: " + stimuli.sizeColumn(0));
        }
        // similar as above, while recording the result and return
        DynamicMatrix layerOutput = stimuli.transpose();
        layerOutput = getWeight(layer).multiplyMtrx(layerOutput);
        layerOutput = layerOutput.transpose();
        layerOutput = layerOutput.plus(getBias(layer));

        // here we already have the weighted and added value of this layer. if the parameter determines to get Z, then we copy the original intermediate value.
        // else, we just return something empty at the position.
        DynamicMatrix ZVal;
        ZVal = new DynamicMatrix();
        return new DynamicMatrix[] {layerOutput, ZVal};
    }

    // ------ feedforward for all layers


    // a forward on all layer, with mapping each layer's output into 0 - 1 using sigmoid function.
    // return: a vector of prediction value.
    public DynamicMatrix feedForward(DynamicMatrix stimuli) {

        // a stimuli must be 1*n pattern.
        if (!stimuli.isVector()) {
            throw new IllegalArgumentException("The stimuli should be 1*n.\n");
        }
        int layerCount = 0;
        while (layerCount != getLayerCount() - 1) {
            stimuli = feedForward(stimuli, layerCount);
            layerCount++;
        }
        return stimuli;
    }

    // a forward on all layer, while storing all intermediate results
    // intermediate result is the calculated value Z, A is mapping Z into 0 - 1 using sigmoid function.
    // returns an array in which: {A, Z}
    // A and Z are list of dynamic matrices.
    public DynamicMatrix[] feedForward(DynamicMatrix stimuli, boolean getZ) {
        if (!stimuli.isVector()) {
            throw new IllegalArgumentException("The stimuli should be 1*n.\n");
        }

        DynamicMatrix[] res = new DynamicMatrix[2], temp;
        res[0] = new DynamicMatrix();
        res[1] = new DynamicMatrix();
        int layerCount = 0;

        while (layerCount != getLayerCount() - 1) {
            temp = feedForward(stimuli, layerCount, getZ);
            res[0].add(temp[0]);
            res[1].add(temp[1]);
            stimuli = temp[0];
            layerCount++;
        }

        return res;
    }

    // a forward on all layer, while storing all intermediate results and do not use sigmoid function.
    // returns an array in which: {A, Z}
    // A and Z are list of dynamic matrices.
    public DynamicMatrix[] feedForwardRaw(DynamicMatrix stimuli, boolean getZ) {
        if (!stimuli.isVector()) {
            throw new IllegalArgumentException("The stimuli should be 1*n.\n");
        }

        DynamicMatrix[] res = new DynamicMatrix[2], temp;
        res[0] = new DynamicMatrix();
        res[1] = new DynamicMatrix();
        int layerCount = 0;

        while (layerCount != getLayerCount() - 1) {
            temp = feedForwardRaw(stimuli, layerCount, getZ);
            res[0].add(temp[0]);
            res[1].add(temp[1]);
            stimuli = temp[0];
            layerCount++;
        }

        return res;
    }

    // a loop of learning, including feed-forward, back-propagation, refresh weight and bias.
    // stimuli is an array of vectors of stimulus, expected is the corresponding expected outcome,
    // eta is the learning rate, batch size should be the number of the stimulus.
    // By default, the function do not do any data process and all training are done with raw data
    // If prefer mapping data within layers to be between 0 - 1 and as results, please use the second function with rawData = true
    public void GDLearning(DynamicMatrix[] stimuli, DynamicMatrix[] expected, double eta, int batchSize) {
        GDLearning(stimuli, expected, eta, batchSize, true);
    }



    // a loop of learning, including feed-forward, back-propagation, refresh weight and bias.
    // stimuli is an array of vectors of stimulus, expected is the corresponding expected outcome,
    // eta is the learning rate, batch size should be the number of the stimulus.
    // if set rawData to be true, then the data will not be processed using sigmoid function to map values into 0 - 1
    public void GDLearning(DynamicMatrix[] stimuli, DynamicMatrix[] expected, double eta, int batchSize, boolean rawData) {

        // an array of matrices, storing A (as the final output of each layer) and Z (as the intermediate result of weighted sum of all nodes in all layers)
        DynamicMatrix[][] AZ = new DynamicMatrix[batchSize][];


        // storage of the errors, which is the adjustment to solve the difference between each output and the expected output
        DynamicMatrix[][] errors = new DynamicMatrix[batchSize][];

        // loop to do the learning
        for (int i = 0; i < batchSize; i++) {

            // a counter
            int layer = getLayerCount() - 1;

            // AZ catches the newest return result of a feedforward.
            if (!rawData) {
                AZ[i] = feedForward(stimuli[i], true);
            }
            else {
                AZ[i] = feedForwardRaw(stimuli[i], true);
            }
            System.out.println("Expected: " + expected[i]);
            System.out.println("Output: " + AZ[i][0].getMtrx(layer-1));

            // from the final layer, back-propagate the errors
            errors[i] = new DynamicMatrix[layer];

            if (!rawData) {
                errors[i][layer - 1] = outputError(AZ[i][0].getMtrx(layer - 1), expected[i], AZ[i][1].getMtrx(layer - 1));
            }
            else {
                errors[i][layer - 1] = outputErrorRaw(AZ[i][0].getMtrx(layer - 1), expected[i], AZ[i][1].getMtrx(layer - 1));
            }
            layer -= 2;
            if (!rawData) {
                while (layer >= 0) {
                    errors[i][layer] = layerError(getWeight(layer + 1), errors[i][layer + 1], AZ[i][1].getMtrx(layer));
                    layer--;
                }
            }
            else {
                while (layer >= 0) {
                    errors[i][layer] = layerErrorRaw(getWeight(layer + 1), errors[i][layer + 1], AZ[i][1].getMtrx(layer));
                    layer--;
                }
            }
        }

        // prepare for the next stage
        int layer = getLayerCount() - 1;
        DynamicMatrix newBias = new DynamicMatrix();
        DynamicMatrix[] errorSums = new DynamicMatrix[layer];
        DynamicMatrix[] outputSums = new DynamicMatrix[layer];
        for (int i = 0; i < layer; i++) {
            errorSums[i] = new DynamicMatrix(1, getNodeCount(i+1), 0);
            outputSums[i] = new DynamicMatrix(1, getNodeCount(i), 0);
        }
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < layer; j++) {
                for (int k = 0; k < errorSums[j].sizeColumn(0); k++) {
                    errorSums[j].set(errorSums[j].get(0, k) + errors[i][j].get(0, k), 0, k);
                }
                for (int k = 0; k < getNodeCount(j); k++) {
                    outputSums[j].set(outputSums[j].get(0, k) + AZ[i][0].get(j, k), 0, k);
                }
            }
        }

        for (int i = 0; i < layer; i++) {
            for (int j = 0; j < getNodeCount(i); j++) {
                errorSums[i].set(errorSums[i].get(0, j) / batchSize, 0, j);
            }
        }

        // as soon as all errors are recorded, update the weights and biases
        for (int j = getLayerCount() - 2; j >= 0; j--) {

            // w = w - eta / batchsize * error * A
            layer = getLayerCount() - 2;
            weights.set(j, weights.get(j).minus((errorSums[j].transpose()).multiplyMtrx(outputSums[j]).multiplyReal(eta / batchSize)));
            newBias.add(bias.getMtrx(j).minus(errorSums[j].multiplyReal(eta / batchSize)), 0);
        }
        bias = newBias;

    }

    // train the model with stimulus and expected values and learning rate
    public void GDlearningRaw(DynamicMatrix[] stimuli, DynamicMatrix[] expected, double eta) {
        GDLearning(stimuli, expected, eta, stimuli.length);
    }

    // train the model with stimulus and expected values and learning rate
    public void GDlearning(DynamicMatrix[] stimuli, DynamicMatrix[] expected, double eta) {
        GDLearning(stimuli, expected, eta, stimuli.length, false);
    }


    // train the model with two files containing stimulus and expected values and learning rate
    // the file processing is handled within this method.
    // Should
    public void GDlearning(File stimuli, File expected, double eta, boolean rawData) {
        try {

            // read the files
            FileReader fr1 = new FileReader(stimuli);
            FileReader fr2 = new FileReader(expected);
            ArrayList<DynamicMatrix> stim = new ArrayList<>();
            ArrayList<DynamicMatrix> exp = new ArrayList<>();

            // a string buffer
            StringBuffer sb;
            int temp = fr1.read();
            while (temp != -1) {

                // skip useless characters in the file
                if ((char)temp == '\n' || (char)temp == '\t' || (char)temp == ' ' || !isVisionableCharacter((char)temp) ) {
                    temp = fr1.read();
                    continue;
                }

                // when met a '{', start buffing the string until all stimulus are recorded
                else if ((char)temp == '{') {

                    // record a stimuli as a matrix
                    sb = new StringBuffer();
                    sb.append('{');
                    temp = fr1.read();
                    while ((char)temp != '}') {
                        sb.append((char)temp);
                        temp = fr1.read();
                    }
                    sb.append('}');

                    // store the matrix
                    stim.add(new DynamicMatrix(sb.toString()));
                    temp = fr1.read();
                }
                else {
                    throw new IllegalArgumentException("Syntax error: " + temp + " found between training sets.\n");
                }
            }

            // similar as above
            temp = fr2.read();
            while (temp != -1) {
                if ((char)temp == '\n' || (char)temp == '\t' || (char)temp == ' ' || !isVisionableCharacter((char)temp)) {
                    temp = fr2.read();
                    continue;
                }
                else if ((char)temp == '{') {
                    sb = new StringBuffer();
                    sb.append('{');
                    temp = fr2.read();
                    while ((char)temp != '}') {
                        sb.append((char)temp);
                        temp = fr2.read();
                    }
                    sb.append('}');
                    exp.add(new DynamicMatrix(sb.toString()));
                    temp = fr2.read();
                }
                else {
                    throw new IllegalArgumentException("Syntax error: " + temp + " found between expected results.\n");
                }
            }
            if (stim.size() != exp.size()) {
                throw new IllegalArgumentException("stimulus counts different from expected result counts.\n");
            }

            DynamicMatrix[] stimArr = new DynamicMatrix[stim.size()];
            DynamicMatrix[] expArr = new DynamicMatrix[exp.size()];

            for (int i = 0; i < stim.size(); i++) {
                stimArr[i] = stim.get(i);
                expArr[i] =exp.get(i);
            }
            if (rawData) {
                GDlearningRaw(stimArr, expArr, eta);
            }
            else {
                GDlearning(stimArr, expArr, eta);
            }
        }
        catch (IOException e) {
            throw new IllegalArgumentException("File not found.\n");
        }
    }


    // ------ basic manipulation methods ------


    public int[] getNodeCount() {
        return nodeCount;
    }

    public int getNodeCount(int layer) {
        return nodeCount[layer];
    }

    public int getLayerCount() {
        return layerCount;
    }

    public int[] size() {
        return nodeCount;
    }

    // get certain weight.
    public double getWeight(int startLayer, int leftNode, int rightNode) {
        if (startLayer >= getLayerCount() - 1 || leftNode >= nodeCount[startLayer] || rightNode >= nodeCount[startLayer+1]) {
            throw new IllegalArgumentException("some of the value out of bound.\n");
        }
        return weights.get(startLayer).get(leftNode, rightNode);
    }

    // get a matrix of weights
    public DynamicMatrix getWeight(int startLayer) {
        return weights.get(startLayer);
    }


    public double getBias(int layer, int node) {
        if (layer >= getLayerCount() || node >= nodeCount[layer]) {
            throw new IllegalArgumentException("some of the value out of bound.\n");
        }
        return bias.get(layer, node);
    }

    public DynamicMatrix getBias(int layer) {
        return bias.getMtrx(layer);
    }

    public DynamicMatrix getBias() {
        return bias;
    }

    public String toString(boolean forFile) {
        if (forFile) {
            StringBuffer sb = new StringBuffer();
            sb.append("(");
            for (int i = 0; i < getLayerCount() - 1; i++) {
                sb.append(getWeight(i));
            }
            sb.append(")\n(");
            for (int i = 0; i < getLayerCount() - 1; i++) {
                sb.append(getBias(i));
            }
            sb.append(")\n");
            return sb.toString();
        }
        else {
            return toString();
        }
    }

    public String toString() {
        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < getLayerCount() - 1; i++) {
            sb.append(i + "->" + (i+1) + ":\n");
            sb.append("weight: \n" + getWeight(i) + "\n");
            sb.append("bias: \n" + getBias(i) + "\n");
        }
        return sb.toString();
    }

    public void toFile(File store) {
        FileWriter fw;
        try {
            fw = new FileWriter(store);
            fw.write(toString(true));
            fw.close();
        }
        catch (IOException e) {
            System.out.println("File not found.\n");
        }
    }

    public void toFile(String fileName) {
        File f1 = new File(DIRNAME, fileName);
        toFile(f1);
    }

    // ------ mathematical methods ------

    // sigmoid function on a single number
    public static double sigmoid(double val) {
        return 1.0 / (1.0 + Math.exp(-val));
    }

    // sigmoid function on a matrix
    public static double[] sigmoid(double[] val) {
        double[] res = new double[val.length];
        for (int i = 0; i < res.length; i++) {
            res[i] = sigmoid(val[i]);
        }
        return res;
    }

    // derivative of sigmoid function
    public static double sigmoidPrime(double val) {
        return sigmoid(val) * (1 - sigmoid(val));
    }

    // derivative of sigmoid function
    public static double[] sigmoidPrime(double[] val) {
        double[] res = new double[val.length];
        for (int i = 0; i < res.length; i++) {
            res[i] = sigmoidPrime(val[i]);
        }
        return res;
    }

    // δL=∇aC⊙σ′(zL);
    // or: δL=(aL−y)⊙σ′(zL), if you prefer.
    // get the errors of the final layer, using the network output, expected output and the intermediate result "Z".
    public static DynamicMatrix outputError(DynamicMatrix outputA, DynamicMatrix expected, DynamicMatrix ZfinalLayer) {
        return (outputA.minus(expected)).hadamardProduct(DynamicMatrix.sigmoidPrime(ZfinalLayer));
    }

    // get the errors of the final layer, using the network output, expected output and the intermediate result "Z".
    public static DynamicMatrix outputErrorRaw(DynamicMatrix outputA, DynamicMatrix expected, DynamicMatrix ZfinalLayer) {
        return (outputA.minus(expected));
    }

    //δl=((wl+1)Tδl+1)⊙σ′(zl)
    // iteratively get the errors of the previous layer using the errors of the next layer and the weights between them.
    public static DynamicMatrix layerError(DynamicMatrix weight, DynamicMatrix errorNextLayer, DynamicMatrix ZThisLayer) {
        return (((weight.transpose()).multiplyMtrx(errorNextLayer.transpose())).transpose()).hadamardProduct(ZThisLayer);
    }

    //δl=((wl+1)Tδl+1)⊙σ′(zl)
    // iteratively get the errors of the previous layer using the errors of the next layer and the weights between them.
    public static DynamicMatrix layerErrorRaw(DynamicMatrix weight, DynamicMatrix errorNextLayer, DynamicMatrix ZThisLayer) {
        return (((weight.transpose()).multiplyMtrx(errorNextLayer.transpose())).transpose()).hadamardProduct(ZThisLayer);
    }


    public static boolean isNumber(char someChar) {
        return ('0' <= someChar && someChar <= '9');
    }

    public static boolean isLetter(char someChar) {
        return ('a' <= someChar && someChar <= 'z' || 'A' <= someChar && someChar <= 'Z');
    }

    public static boolean isVisionableCharacter(char someChar) {
        return (32 <= someChar && someChar <= 127);
    }


    // ------ main ------

    public static void main(String[] Args) {

    /*
        MnistReader mnr = new MnistReader();


        Scanner s = new Scanner(System.in);
        System.out.println("Please input the name of the network data file: ");
        String f2 = s.next();

        String stim = "stim.txt";
        String exp = "exp.txt";
        Network newNet = new Network(new File(DIRNAME, f2));

        System.out.println(newNet);

        int i = 0;
        while (i < 100) {
            newNet.GDlearning(new File(DIRNAME, stim), new File(DIRNAME, exp), 0.00001, true);
            i++;
        }



        System.out.println(newNet.feedForwardRaw(new DynamicMatrix(new double[][] {{2015}}), true)[0].get(0,0));
    */
        MnistReader mnr = new MnistReader();


        Scanner s = new Scanner(System.in);
        System.out.println("Please input the name of the network data file: ");
        String f2 = s.next();

        String stim = "stim1.txt";
        String exp = "exp1.txt";
        Network newNet = new Network(new File(DIRNAME, f2));

        System.out.println(newNet);

        int i = 0;
        while (i < 100) {
            newNet.GDlearning(new File(DIRNAME, stim), new File(DIRNAME, exp), 0.2, false);
            i++;
        }



        System.out.println(newNet.feedForwardRaw(new DynamicMatrix(new double[][] {{1,1}}), true)[0].get(0,0));




    }


}
