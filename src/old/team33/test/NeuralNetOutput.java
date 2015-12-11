package old.team33.test;

import java.io.IOException;

import old.team33.NeuralNetwork;
import net.razorvine.pickle.PickleException;

public class NeuralNetOutput {

    public static void main(String[] args) throws PickleException, IOException {
    	NeuralNetwork nn = new NeuralNetwork(true);
//    	double[] data = new double[]{0.00969284, 2032.56, 942.478, -0.000612781, 0.00284858, -0.000184746, 4.99776, 5.20369, 5.8866, 7.46181, 11.9366, 200.0, 136.129, 74.6482, 51.6945, 39.7687, 32.5149, 27.6773, 24.0663, 21.1567, 16.9484, 13.2321, 11.2589, 10.2851, 10.0029, 0.333666, 0.0, 0.0, 0.0, 0.0, 0.345609, 0.0, 0.0, 0.0, 0.0, 0.0};
    	int n_inputs = 26;
    	double[] data = new double[n_inputs];
    	for(int i=0;i<n_inputs;i++){data[i] = 0.2;}
    	double output = nn.getSteering(data);
    	
    	System.out.println("NN steering output: "+output);
    }

}