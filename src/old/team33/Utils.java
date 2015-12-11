package old.team33;

import java.util.ArrayList;

public class Utils {
	
    public double[][] set_matrix_value(double[][] matrix, int x, int y, double value) {
    	/*
    	 * Dynamically reshapes the matrix
    	 */
    	
        if (x >= matrix.length) {
            double[][] tmp = matrix;
            matrix = new double[x + 1][];
            System.arraycopy(tmp, 0, matrix, 0, tmp.length);
            for (int i = x; i < x + 1; i++) {
                matrix[i] = new double[y];
            }
        }

        if (y >= matrix[x].length) {
            double[] tmp = matrix[x];
            matrix[x] = new double[y + 1];
            System.arraycopy(tmp, 0, matrix[x], 0, tmp.length);
        }

        matrix[x][y] = value;
        return matrix;
    }
    
	public double[][] toPrimitive(ArrayList<ArrayList<Double>> matrix_double){
		double[][] conv = new double[1][1];
		int n_rows = 0;
		int n_cols;
		for(ArrayList<Double> row:matrix_double){
			n_cols = 0;
			for(Double ele:row){
				//System.out.println(ele);
				//conv[n_rows][n_cols] = ele;
				conv = this.set_matrix_value(conv, n_rows, n_cols, ele);
				n_cols++;
			}
			n_rows++;
		}
		return conv;
	}
	
	public double[] toPrimitiveVector(ArrayList<Double> vector_double){
		double[] conv = new double[1];
		int pos = 0;
//		for(ArrayList<Double> ele_array:vector_double){
		for(Double ele:vector_double){
//			Double ele = ele_array.get(0);
			int len = conv.length;
			if (pos >= len){
				double[] tmp = conv;
				conv = new double[len+1];
	            System.arraycopy(tmp, 0, conv, 0, tmp.length);
			}
            conv[pos] = ele;
            pos++;
		}
		
		return conv;
	}

}
