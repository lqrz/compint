package old.team33.example;

import old.team33.FFNetwork;
import old.team33.NeuralNetwork;

import org.encog.neural.networks.BasicNetwork;

import team33.NNPredictor;
import cicontest.torcs.genome.IGenome;

public class DefaultDriverGenome implements IGenome {
    
	private static final long serialVersionUID = 6534186543165341653L;
    
//    private NeuralNetwork myNN = new NeuralNetwork(10,8,2);
//	private NeuralNetwork nn = new NeuralNetwork(true);
	private NNPredictor nn = new NNPredictor("PATH_TO_FILE_HERE");
	private NNPredictor nn_accel = new NNPredictor("PATH_TO_FILE_HERE");
    private String[] inputfields = InputReader.ReadFile("PATH_TO_FILE_HERE");
	
    public NNPredictor getNNsteer() {
        return this.nn;
    }
    
    public NNPredictor getNNaccel() {
        return this.nn_accel;
    }
    
    public String[] getInputFields() {
    	return inputfields;
    }
}

