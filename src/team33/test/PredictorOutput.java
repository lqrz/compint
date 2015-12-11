package team33.test;

import java.io.IOException;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import team33.NNPredictor;

public class PredictorOutput {

    public static void main(String[] args) throws IOException {
    	NNPredictor nn = new NNPredictor("trained_files/theano_mlp/steering_test.txt");
    	double[] data_d = new double[]{0.12057883,0.69185077,1.02532707,0.93692504,-0.0454854,-0.35208915,-0.09842254,-0.55460677,-0.58100937,-0.06733024,-0.25887767,-0.33820493
    			,-0.2137024,0.2916204,4.91157347,1.2735095,0.27715893,-0.2930171,-0.44367666,-0.43055595,-0.29202147,-0.31645138,-0.38450405,-1.21369013,-0.02400128,-0.04230797,1.00900087,1.0086114,1.00577571,0.98914236,-0.50434044,0.,0.,0.,0.,0.};
//    	int n_inputs = 26;
//    	double[] data = new double[n_inputs];
//    	for(int i=0;i<n_inputs;i++){data[i] = 0.2;}
    	RealMatrix data = new Array2DRowRealMatrix(data_d);
    	double output = nn.predict(data);
    	
    	System.out.println("NN steering output: "+output);
    }

}