package old.team33;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import net.razorvine.pickle.PickleException;
import net.razorvine.pickle.Unpickler;

import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;



public class ESN {
	
	public ESN() {
		super();
		// TODO Auto-generated constructor stub
	}

	public static RealMatrix random(int M, int N, double sparsity) {
        double[][] A = new double[M][N];
        double threshold;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
        		threshold = Math.random();
        		if (threshold > sparsity) {
        			A[i][j] = Math.random();
				} else {
					A[i][j] = 0.0;
				}
            }
        }
        return MatrixUtils.createRealMatrix(A);
    }
	
	public static RealMatrix random(int M, int N) {
        return random(M, N, 1.0);
    }


    public static void main(String[] args) {
    	
    	double[][] x = new double[1][1];
    	double[][] d = new double[1][1];
    	
    	/*
    	List<String> sensor_features = Arrays.asList("angle", "curLapTime", "damage","distFromStart","distRaced","fuel","gear","lastLapTime",
    			"opponents","racePos","rpm","speedX","speedY","speedZ","track","trackPos","wheelSpinVel","z","focus");
    	
    	List<String> action_features = Arrays.asList("accel","brake","clutch","gear","steer","meta","focus");
		*/
    	
    	List<String> sensor_features = Arrays.asList("angle", "gear","rpm","speedX","speedY","speedZ","track","trackPos","wheelSpinVel","z","focus");
    	
    	List<String> action_features = Arrays.asList("steer");
    	
    	//String path = "c:\\Users\\lqrz\\Documents\\Lautaro\\AI\\ci\\champ2010client-java\\classes\\log.txt";
    	String path = "/Users/fbuettner/dev/UvA/compint/1lap.log";
    	
    	int i = 0;
    	try(BufferedReader br = new BufferedReader(new FileReader(path))) {
    	    for(String line; (line = br.readLine()) != null; ) {
    	    	
    	    	if((i+1)%1000==0) System.out.println(i+1+" lines processed");
    	    		
    	        // process the line.
    	    	//System.out.println(line);
    	    	
    	    	//List<Double> feats = new ArrayList<Double>();
	    		//Double[] vals_doubles;
    	    	if (Pattern.compile("^Received: \\(.*").matcher(line).matches()){
		    		int j = 0;
	    	    	for(String sensor_feat : sensor_features){
		    	    	Matcher m = Pattern.compile("\\("+sensor_feat+"\\s(-?\\d*(\\.)?\\d*(E-|e-)?\\d*+(\\s)?)*\\)").matcher(line);
		    	    	while(m.find()){
		    	    		//System.out.println(m.group(0));
		    	    		//System.out.println(m.group(0).replace("\\(angle ((\\d\\.(\\d)*(\\s)?)*)\\)", "$1"));
		    	    		//System.out.println(m.group(0).replaceAll("\\("+sensor_feat+"\\s((-?\\d*(\\.)?(\\d)+(\\s)?)*)\\)", "$1"));
		    	    		String[] vals_string = m.group(0).replaceAll("\\("+sensor_feat+"\\s((-?\\d*(\\.)?\\d*(E-|e-)?\\d*+(\\s)?)*)\\)", "$1").split("\\s");
		    	    		//ArrayList<Double> vals_double = new ArrayList<Double>() {{for (String v : vals_string) add(new Double(v));}};
		    	    		//feats.addAll(vals_double);
		    	    		for (String v : vals_string){
		    	    			//vals_doubles[j]=new Double(v);
		    	    			//matrix[i][j] = new Double(v);
		    	    			x = set_matrix_value(x,i,j,new Double(v));
		    	    			j++;
		    	    		}
		    	    	}
	    	    	}
	    	    	//check that all samples have the same number of features
	    	    	if (x[i].length != x[0].length){
	    	    		System.out.println("line "+(i+1)+" "+x[0].length+" "+x[i].length);
	    	    		for(double v:x[i]){System.out.print(v+" ");}
	    	    		System.exit(0);
	    	    	}
    	    	} else if(Pattern.compile("^Sending: \\(.*").matcher(line).matches()){
		    		int j = 0;
	    	    	for(String action_feat : action_features){
		    	    	Matcher m = Pattern.compile("\\("+action_feat+"\\s(-)?\\d*(\\.)?(\\d)*(E-|e-)?(\\d)*\\)").matcher(line);

		    	    	while(m.find()){
		    	    		//System.out.println(m.group(0).replaceAll("\\("+action_feat+"\\s(-?\\d*(\\.)?(\\d)+)\\)", "$1"));		    	    	}
		    	    		String val_string = m.group(0).replaceAll("\\("+action_feat+"\\s((-)?\\d*(\\.)?(\\d)*(E-|e-)?(\\d)*)\\)", "$1");
		    	    		//System.out.println(new Double(val_string));
		    	    		d = set_matrix_value(d,i,j,new Double(val_string));
		    	    		j++;
		    	    	}
	    	    	}
	    	    	i++;
    	    	}
    	    }
    	    // line is not visible here.
    	}catch(IOException e){
    		System.out.println("ERROR while opening file");
    		e.printStackTrace();
    	}
    	
    	//print_matrix(d);
    	System.out.println("construct matrices");
    	RealMatrix x_mat = MatrixUtils.createRealMatrix(x);
    	RealMatrix d_mat = MatrixUtils.createRealMatrix(d);
    	
    	
    	int n_samples = x_mat.getRowDimension();
    	int n_features = x_mat.getColumnDimension();
    	int n_hidden = (int) (n_samples / 10);
    	//int n_hidden = 10;
    	int n_out = 1;
    	
    	//double[][] w_in = new double[n_hidden][n_features];
    	//double[][] w = new double[n_hidden][n_hidden];
    	//double[][] w_out = new double[n_hidden][n_out];
    	//double[][] w_back = new double[n_hidden][n_out];
    	
    	double alpha = 0.5; //hyperparam
    	RealMatrix w_mat = random(n_hidden, n_hidden, 0.99);
    	RealMatrix w_in_mat = random(n_hidden, n_features);
    	RealMatrix w_out_mat = random(n_hidden, n_out);
    	RealMatrix w_back_mat = random(n_hidden, n_out);
//    	Matrix w_mat = Matrix.random(n_hidden, n_hidden);
//    	Matrix w_in_mat = Matrix.random(n_hidden, n_features);
//    	Matrix w_out_mat = Matrix.random(n_hidden, n_out);
//    	Matrix w_back_mat = Matrix.random(n_hidden, n_out);
    	//w_mat.print(2, 2);

    	
    	System.out.println("compute Eigenvalues");
    	EigenDecomposition eig = new EigenDecomposition(w_mat);
    	double scale = eig.getRealEigenvalue(0);
    	System.out.println(scale);
    	w_mat  = w_mat.scalarMultiply(alpha/scale);
   	
    	int t_0 = 100; //washout
    	int sampling_iters = 200; //nr of iters to sample
    	System.out.println("fill in samples");
    	double[][] m = new double[sampling_iters][n_hidden];
    	double[][] t = new double[sampling_iters][1];
		RealMatrix x_n = MatrixUtils.createRealMatrix(new double[n_hidden][1]);
    	
		// Matrix x_n_mat;
		
    	for(int it=0; it<sampling_iters; it++){
    		//if(it!=0){
			// x_n_mat = new Matrix(x_n,1).transpose();
			
			//get input
			//double[] vec = x_mat.getArray()[it];
			//Matrix vec_mat = new Matrix(vec,1).transpose();
			//vec_mat.print(2, 2);
			
			//get true output
			RealMatrix dd = MatrixUtils.createRealMatrix(new double[][]{d[it]});
			double[][] xx = new double[][]{x[it]};
			//System.out.println("w_back_mat "+w_back_mat.getRowDimension()+" "+w_back_mat.getColumnDimension());
			
			//update x_n
			RealMatrix xx_mat = MatrixUtils.createRealMatrix(xx);
			System.out.println("xx_mat shape: "+xx_mat.getRowDimension()+"x"+xx_mat.getColumnDimension());
			System.out.println("w_in_mat shape: "+w_in_mat.getRowDimension()+"x"+w_in_mat.getColumnDimension());
			System.out.println("w_mat shape: "+w_mat.getRowDimension()+"x"+w_mat.getColumnDimension());
			System.out.println("x_n shape: "+x_n.getRowDimension()+"x"+x_n.getColumnDimension());
			System.out.println("w_back_mat shape: "+w_back_mat.getRowDimension()+"x"+w_back_mat.getColumnDimension());
			System.out.println("dd shape: "+dd.getRowDimension()+"x"+dd.getColumnDimension());			
			
			x_n = w_in_mat.multiply(xx_mat.transpose()).add(w_mat.multiply(x_n)).add(w_back_mat.multiply(dd));
    		//}
			
			//if not in washout, store m and t
    		if (it > t_0){
    			
    			//System.out.println(x_n_mat.getRowDimension()+" "+x_n_mat.getColumnDimension());
    			
    			//store m
    			m[it-t_0] = x_n.getColumn(0);
    			
    			//store t
    			t[it-t_0][0] = dd.getEntry(0, 0);
    		}
    	}

    	System.out.println("Computing inverse of w and multiply with t");
    	
    	//System.out.println("m_inverse "+m_inverse.getRowDimension()+" "+m_inverse.getColumnDimension());
    	//System.out.println("t "+new Matrix(t,1).getRowDimension()+" "+new Matrix(t,1).getColumnDimension());
    	//MatrixUtils.inverse(matrix)
    	w_out_mat = MatrixUtils.inverse(MatrixUtils.createRealMatrix(m)).multiply(MatrixUtils.createRealMatrix(t));
    	//System.out.println("w_out_mat "+w_out_mat.getRowDimension()+" "+w_out_mat.getColumnDimension());
    	
    	PrintWriter writer = null;
    	try {
    	    writer = new PrintWriter("w_out.txt", "UTF-8");
    	    writer.write(w_out_mat.toString());
    	    writer.flush();
    	} catch (FileNotFoundException | UnsupportedEncodingException e) {
    	    // TODO Auto-generated catch block
    	    e.printStackTrace();
    	}
    }

    private static double[][] set_matrix_value(double[][] matrix, int x, int y, double value) {
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
    
    private static void print_matrix(double[][] matrix){
    	for (int i = 0; i < matrix.length; i++) {
    	    for (int j = 0; j < matrix[0].length; j++) {
    	    	try{
    	    		System.out.print(matrix[i][j] + " ");
    	    	}catch(NullPointerException e){
    	    		System.out.println("ERRO");
    	    	}
    	    }
    	    System.out.print("\n");
    	}
    }
    
    @SuppressWarnings("unchecked")
	public ArrayList<ArrayList<Double>> loadPickleMatrix(String path) throws PickleException, IOException{

    	InputStream stream = new FileInputStream(path);
    	
    	Unpickler unpickler = new Unpickler();
    	ArrayList<ArrayList<Double>> data = (ArrayList<ArrayList<Double>>) unpickler.load(stream);
    	
    	return data;
    }

    @SuppressWarnings("unchecked")
	public ArrayList<Double> loadPickleVector(String path) throws PickleException, IOException{

    	InputStream stream = new FileInputStream(path);
    	
    	Unpickler unpickler = new Unpickler();
    	ArrayList<Double> data = (ArrayList<Double>) unpickler.load(stream);
    	
    	return data;
    }
	
}