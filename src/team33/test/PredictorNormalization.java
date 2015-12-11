package team33.test;

import java.io.IOException;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import team33.NNPredictor;

public class PredictorNormalization {

    public static void main(String[] args) throws IOException {
    	NNPredictor nn = new NNPredictor("trained_files/theano_mlp/steering_test.txt");
    	double[] data_d = new double[]{-1.74846000e-07,0.00000000e+00,9.42478000e+02,0.00000000e+00,0.00000000e+00,1.45926000e-03,6.27322000e+00,4.91358000e+00,7.82592000e+00,1.11684000e+01,1.27865000e+01,1.57804000e+01,1.65436000e+01,2.84594000e+01,5.54406000e+01,1.57316000e+02,1.05106000e+02,4.78536000e+01,3.56892000e+01,3.18112000e+01,2.22805000e+01,2.21989000e+01,1.45743000e+01,1.20983000e+01,9.29892000e+00,3.33331000e-01,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,3.45259000e-01,-1.00000000e+00,-1.00000000e+00,-1.00000000e+00,-1.00000000e+00,-1.00000000e+00};
//    	int n_inputs = 26;
//    	double[] data = new double[n_inputs];
//    	for(int i=0;i<n_inputs;i++){data[i] = 0.2;}
    	RealMatrix data = new Array2DRowRealMatrix(data_d);
    	double output = nn.predict_normalization(data, true);
    	
    	System.out.println("NN steering output: "+output);
    }

}