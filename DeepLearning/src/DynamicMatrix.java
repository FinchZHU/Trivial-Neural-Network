import java.io.EOFException;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * Created by admin on 2018/2/9.
 */
public class DynamicMatrix {

    // a matrix with arbitrary length of rows and columns
    //
    //                   columns
    //      {(0,0) (0,1) ...      (0,y1)}
    //      {(1,0) (1,1) ...      (1,y2)}
    //      {(2,0)       ...      ...   }
    // rows {(3,0)       .        ...   }
    //      {(4,0)        .       ...   }
    //      {...           .      ...   }
    //      {(x,0)       ...      (x,yx)}



    private ArrayList<ArrayList<Double>> data;


    // eigenvalue
    // eigen vector?
    // diagonal
    // orthogonal diagonal
    // and more! ... good luck.



    // ------ constructors ------


    // a matrix. several types of constructor here
    public DynamicMatrix(double[][] data) {

        this.data = new ArrayList<>();

        for (int i = 0; i < data.length; i++) {
            this.data.add(new ArrayList<>());
            for (int j = 0; j < data[i].length; j++) {
                this.data.get(i).add(data[i][j]);
            }
        }

    }

    // construct a matrix in the form: {[x11, x12, x13 ... x1n1]
    //                                  [x21, x22, x23 ... x2n2]
    //                                  ...
    //                                  [xn1, xn2, xn3 ... xnnn]}
    public DynamicMatrix(String data) {
        this();

        StringBuffer sb = new StringBuffer(data);
        int count = 0;

        while (sb.charAt(count) == ' ' || sb.charAt(count) == '\n' || sb.charAt(count) == '\t') {
            count++;
        }

        char tempChar = sb.charAt(count);
        if (tempChar != '{') {
            throw new IllegalArgumentException("Syntax incorrect: matrix starts not with a '{', but " + tempChar + " instead. count: " + count + "\n");
        }

        // a mark indicating which layer the reader currently is in, row layer or column layer.
        boolean betweenRows = true, rowHasNumber = false;

        // counters of rows and columns of each row.
        int rowCount = 0;
        ArrayList<Integer> columnCount = new ArrayList();

        // a workplace recording the reading number without calculating the digits after points. Since the file is read byte by byte, the number will be read iteratively rather than as a whole.
        int number = 0;

        // the real value that is finally put into matrix.
        double value = 0.0;

        // an indicator and a counter, indicating the occurrence of point and counting how many digits have been gone through after the point.
        boolean pointed = false, negative = false;
        int afterPoint = 0;

        // loop until the end of the matrix data
        while ((tempChar = (sb.charAt(count))) != '}') {

            count++;
            // ignore empty characters
            if (tempChar == ' ' || tempChar == '\n' || tempChar == '\t' || !Network.isVisionableCharacter(tempChar)) {
                continue;
            }

            // if met [, start a new row
            if (tempChar == '[') {
                if (!betweenRows) {
                    throw new IllegalArgumentException("Syntax incorrect: 2 '[' appears consecutively without a ']' in between. Location: (" + rowCount + ", " + columnCount.get(rowCount) + ").\n");
                }
                betweenRows = false;
                rowHasNumber = false;
                add(new DynamicMatrix());
                columnCount.add(0);
                continue;
            }

            // if met ], end row
            if (tempChar == ']') {
                if (betweenRows) {
                    throw new IllegalArgumentException("Syntax incorrect: 2 ']' appears consecutively without a '[' in between. Location: (" + rowCount + ", " + columnCount.get(rowCount) + ").\n");
                }
                if (rowHasNumber) {
                    value = number / Math.pow(10, afterPoint);
                    if (negative) {
                        add(-value, rowCount, columnCount.get(rowCount));
                    }
                    else {
                        add(value, rowCount, columnCount.get(rowCount));
                    }
                    columnCount.set(rowCount, columnCount.get(rowCount) + 1);
                    pointed = false;
                    afterPoint = 0;
                    number = 0;
                    value = 0;
                    negative = false;
                }
                betweenRows = true;
                rowCount++;
                continue;
            }

            // if met comma, store the number
            if (tempChar == ',') {
                value = number / Math.pow(10, afterPoint);
                if (negative) {
                    add(-value, rowCount, columnCount.get(rowCount));
                }
                else {
                    add(value, rowCount, columnCount.get(rowCount));
                }
                columnCount.set(rowCount, columnCount.get(rowCount) + 1);
                pointed = false;
                afterPoint = 0;
                number = 0;
                value = 0;
                negative = false;
                continue;
            }

            // if met number, refresh the number
            if ('0' <= tempChar && tempChar <= '9') {
                if (pointed) {
                    afterPoint++;
                }
                rowHasNumber = true;
                number *= 10;
                number += (tempChar - '0');
                continue;
            }

            // if met point, start counting how many digits are after the point.
            if (tempChar == '.') {
                pointed = true;
            }
            if (tempChar == '-') {
                negative = true;
            }
        }
    }


    // create a matrix with data in a file.
    // the data should in form: {[x11, x12, x13 ...][x21, x22, x23...]...[xn1, xn2, xn3...]}
    public DynamicMatrix(File file) {
        this(toString(file));
    }

    // initialize a matrix with all initial value in rows and columns of each row
    public DynamicMatrix(int rowCount, int[] columnCount, double initVal) {

        this.data = new ArrayList<>();

        for (int i = 0; i < rowCount; i++) {
            this.data.add(new ArrayList<>());
            for (int j = 0; j < columnCount[i]; j++) {
                this.data.get(i).add(initVal);
            }
        }

    }

    // initial with initial value 0
    public DynamicMatrix(int rowCount, int[] columnCount) {
        this(rowCount, columnCount, 0);
    }

    // initial a standard matrix with value
    public DynamicMatrix(int rowCount, int columnCount, double initVal) {
        this(rowCount, getIntArr(rowCount, columnCount), initVal);
    }

    public DynamicMatrix(int rowCount, int columnCount) {
        this(rowCount, columnCount, 0);
    }

    public DynamicMatrix() { this(0, 0); }

    // ------ mathematical methods ------

    // matrix plus another matrix
    // mathematical
    public DynamicMatrix plus(DynamicMatrix other) {
        if (!isRealMatrix() || !other.isRealMatrix()) {
            throw new IllegalArgumentException("Not both of the matrices are standard matrix.\n");
        }
        if (sizeRow() != other.sizeRow() || sizeColumn(0) != other.sizeColumn(0)) {
            throw new IllegalArgumentException("Size not match, " +
                    "this matrix: {" + sizeRow() + ":" + sizeColumn(0) +"}, " +
                    "other matrix: {" + other.sizeRow() + ":" + other.sizeColumn(0) + "}.\n");
        }
        DynamicMatrix res = new DynamicMatrix(new double[0][0]);
        for (int i = 0; i < sizeRow(); i++) {
            for (int j = 0; j < sizeColumn(i); j++) {
                res.add(get(i,j) + other.get(i,j), i, j);
            }
        }
        return res;
    }


    // see if something is a vector.
    public boolean isVector() {
        return (sizeRow() == 1);
    }

    // see if two matrices are in the same size.
    public boolean isEquallyLarge(DynamicMatrix other) {
        int rowCount = 0;
        if (sizeRow() != other.sizeRow()) {
            return false;
        }
        while (rowCount < sizeRow()) {
            if (sizeColumn(rowCount) != other.sizeColumn(rowCount)) {
                return false;
            }
            rowCount++;
        }
        return true;
    }


    // that's just minus in a matrix.
    public DynamicMatrix minus(DynamicMatrix other) {
        if (!isRealMatrix() || !other.isRealMatrix()) {
            throw new IllegalArgumentException("Not both of the matrices are standard matrix.\n");
        }
        if (sizeRow() != other.sizeRow() || sizeColumn(0) != other.sizeColumn(0)) {
            throw new IllegalArgumentException("Size not match, " +
                    "this matrix: {" + sizeRow() + ":" + sizeColumn(0) +"}, " +
                    "other matrix: {" + other.sizeRow() + ":" + other.sizeColumn(0) + "}.\n");
        }
        DynamicMatrix res = new DynamicMatrix(new double[0][0]);
        for (int i = 0; i < sizeRow(); i++) {
            for (int j = 0; j < sizeColumn(i); j++) {
                res.add(get(i,j) - other.get(i,j), i, j);
            }
        }
        return res;
    }

    // the hadamard product returns a new matrix with each element a product of the two elements in separate original matrices of the same location.
    public DynamicMatrix hadamardProduct(DynamicMatrix other) {
        if (!this.isVector() || !other.isVector() || !this.isEquallyLarge(other)) {
            throw new IllegalArgumentException("The two matrices are not all vectors or not equally long, " +
                    "this matrix: {" + sizeRow() + ":" + sizeColumn(0) + "}, " +
                            "other matrix: {" + other.sizeRow() + ":" + other.sizeColumn(0) + "}.\n");
        }
        DynamicMatrix res = new DynamicMatrix();
        for (int i = 0; i < sizeColumn(0); i++) {
            res.add(get(0, i) * other.get(0, i), 0, i);
        }
        return res;
    }

    // matrix multiplication;
    // it works. Yay.
    public double[][] multiplyArr(DynamicMatrix other) {

        // check if the two matrices are multipliable.
        if (!other.isRealMatrix() || !isRealMatrix()) {
            throw new IllegalArgumentException("Is not real matrix. \n");
        }
        if (sizeColumn(0) != other.sizeRow()) {
            throw new IllegalArgumentException("Size not match, " +
                    "this matrix: {" + sizeRow() + ":" + sizeColumn(0) +"}, " +
                    "other matrix: {" + other.sizeRow() + ":" + other.sizeColumn(0) + "}.\n");
        }

        // create a workplace to put result in.
        int resX = sizeRow(), resY = other.sizeColumn(0);
        double[][] result = new double[resX][resY];

        // iterate the whole matrix and calculate result[i][j] one by one.
        for (int i = 0; i < resX; i++) {
            for (int j = 0; j < resY; j++) {
                for (int leftj = 0; leftj < sizeColumn(0); leftj++) {
                    result[i][j] += get(i, leftj) * other.get(leftj, j);
                }
            }
        }
        return result;

    }

    // multiplication of Dynamic matrix version.
    public DynamicMatrix multiplyMtrx(DynamicMatrix other) {
        DynamicMatrix result = new DynamicMatrix(this.multiplyArr(other));
        return result;
    }

    // multiply with a real number
    public DynamicMatrix multiplyReal(double real) {

        DynamicMatrix res = new DynamicMatrix();
        double val;

        for (int i = 0; i < sizeRow(); i++) {
            for (int j = 0; j < sizeColumn(i); j++) {
                val = get(i, j);
                res.add(val * real, i, j);
            }
        }

        return res;
    }



    // ------ Manipulation methods ------


    // basic
    // get the number of rows
    public int sizeRow() {
        return data.size();
    }

    // basic
    // get the number of columns of rowNum
    public int sizeColumn(int rowNum) {
        if (rowNum >= sizeRow()) {
            return 0;
        }
        return data.get(rowNum).size();
    }

    // get a transpose of a standard matrix
    public DynamicMatrix transpose() {
        if (!isRealMatrix()) {
            throw new IllegalArgumentException();
        }
        DynamicMatrix result = new DynamicMatrix(sizeColumn(0), sizeRow());
        for (int i = 0; i < sizeColumn(0); i++) {
            for (int j = 0; j < sizeRow(); j++) {
                result.set(get(j, i), i, j);
            }
        }

        return result;
    }

    // get the numbers of columns in all rows
    public int[] sizeColumn() {
        int[] sizeCol = new int[sizeRow()];
        for (int i = 0; i < sizeCol.length; i++) {
            sizeCol[i] = sizeColumn(i);
        }
        return sizeCol;
    }


    // see if all rows have the same number of columns
    public boolean isRealMatrix() {

        int[] sizeArr = sizeColumn();
        int standard = sizeArr[0];
        for (int i = 0; i < sizeRow(); i++) {
            if (sizeArr[i] != standard) {
                return false;
            }
        }
        return true;
    }

    // basic
    // get X_(rowX,columnY)
    public double get(int rowX, int columnY) {
        if (rowX >= data.size() || columnY >= data.get(rowX).size()) {
            throw new IllegalArgumentException("rowX: " + rowX + ", rowY: " + columnY);
        }
        return data.get(rowX).get(columnY);
    }

    // return an array of value.
    public double[] getArr(int rowX) {
        double[] vector = new double[sizeColumn(rowX)];
        for (int i = 0; i < sizeColumn(rowX); i++) {
            vector[i] = get(rowX, i);
        }
        return vector;
    }

    // get the double form of the matrix
    public double[][] getArr() {
        if (!isRealMatrix()) {
            throw new IllegalArgumentException("Not standard.\n");
        }
        double[][] matrix = new double[sizeRow()][sizeColumn(0)];
        for (int i = 0; i < matrix.length; i++) {
            matrix[i] = getArr(i);
        }
        return matrix;
    }

    // get a vector of Dynamic-matrix
    public DynamicMatrix getMtrx(int rowX) {
        double[] result = getArr(rowX);
        double[][] resource = new double[1][result.length];
        resource[0] = result;
        return new DynamicMatrix(resource);
    }

    // basic
    // set X_(rowX,columnY) to be value
    public void set(double value, int rowX, int columnY) {
        if (rowX >= sizeRow() || columnY >= getArr(rowX).length) {
            throw new IllegalArgumentException();
        }

        data.get(rowX).set(columnY, value);
    }



    // basic
    // set X_(rowX,columnY) to be value
    // the matrix method plus (not the mathematical one)
    public void add(double value, int rowX, int columnY) {
        if (rowX >= sizeRow() + 1 || columnY >= sizeColumn(rowX) + 1) {
            throw new IllegalArgumentException();
        }
        if (rowX == sizeRow()) {
            data.add(new ArrayList<>());
        }
        if (columnY == sizeColumn(rowX)) {
            data.get(rowX).add(value);
        }
        else {
            data.get(rowX).add(columnY, value);
        }
    }

    // add a matrix into this one.
    public void add(DynamicMatrix newRow, int rowX) {
        if (rowX >= sizeRow()+1) {
            throw new IllegalArgumentException("Row number out of bound.\n");
        }
        for (int i = rowX; i < rowX + newRow.sizeRow(); i++) {
            data.add(i, new ArrayList<>());
            for (int j = 0; j < newRow.sizeColumn(i - rowX); j++) {
                add(newRow.get(i - rowX, j), i, j);
            }
        }

    }

    // add a matrix at the end.
    public void add(DynamicMatrix newRow) {
        add(newRow, sizeRow());
    }


    // delete some value.
    public void delete(int rowX, int columnY) {
        if (rowX >= data.size() || columnY >= data.get(rowX).size()) {
            throw new IllegalArgumentException();
        }
        data.get(rowX).remove(columnY);
    }


    // delete a whole row. (Are you sure?)
    public void delete(int rowX) {
        if (rowX >= data.size()) {
            throw new IllegalArgumentException();
        }
        data.remove(rowX);
    }

    // delete all data.
    public void erase() {
        for (int i = sizeRow()-1; i >=0; i--) {
            delete(i);
        }
    }




    // ------ static and trivial methods ------

    // static method to create an array of size-number of initVal, return it.
    public static int[] getIntArr(int size, int initVal) {

        int[] arr = new int[size];
        Arrays.fill(arr, initVal);

        return arr;
    }

    public static String toString(File file) {
        try {
            // start reading
            FileReader fr = new FileReader(file);

            StringBuffer sb = new StringBuffer();
            int temp = fr.read();
            while (temp != -1) {
                sb.append((char)temp);
                temp = fr.read();
            }
            fr.close();
            return sb.toString();
        }
        catch (IOException e) {
            throw new IllegalArgumentException("File not found.\n");
        }
    }


    // for 1-dim matrices
    public static DynamicMatrix sigmoid(DynamicMatrix val) {
        if (!val.isVector()) {
            throw new IllegalArgumentException("This is not a vector.\n");
        }
        double[] value = val.getArr(0);
        value = Network.sigmoid(value);
        double[][] resource = new double[1][value.length];
        resource[0] = value;
        return new DynamicMatrix(resource);
    }

    public static DynamicMatrix sigmoidPrime(DynamicMatrix val) {
        if (!val.isVector()) {
            throw new IllegalArgumentException("This is not a vector.\n");
        }
        double[] value = val.getArr(0);
        value = Network.sigmoidPrime(value);
        double[][] resource = new double[1][value.length];
        resource[0] = value;
        return new DynamicMatrix(resource);
    }

    // return a string of the whole matrix.
    public String toString() {
        StringBuffer sb = new StringBuffer("{");
        for (int i = 0; i < sizeRow(); i++) {
            sb.append("[");
            for (int j = 0; j < sizeColumn(i); j++) {
                sb.append(get(i, j));
                sb.append(", ");
            }
            sb.deleteCharAt(sb.length()-1);
            sb.deleteCharAt(sb.length()-1);
            sb.append("]\n");
        }
        if (sb.charAt(sb.length()-1) == '\n') {
            sb.deleteCharAt(sb.length()-1);
        }
        sb.append("}");

        return sb.toString();
    }


    // ------ main method ------

    public static void main(String[] Args) {

        // double[][] data1 = new double[][] {{1,-2,1},{-1,4,-3},{1,-1,1}};
        // double[][] data = new double[][] {{-3.75,-0.25,5},{1.5,-0.5,-4},{7.25,1.75,-5}};

        double[][] data1 = new double[][] {{0.2162 * 1930}};
        double[][] data = new double[][] {{-355.1834}};


        DynamicMatrix mat1 = new DynamicMatrix(data1);
        DynamicMatrix mat = new DynamicMatrix(data);
        mat.add(mat1);
        System.out.println(mat);
    }


}
