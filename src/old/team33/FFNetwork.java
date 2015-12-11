package old.team33;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.util.obj.SerializeObject;


public class FFNetwork {
	
	private static String outputFileName = "trained_files/FFN_steering.mem";

	public FFNetwork() {
		// TODO Auto-generated constructor stub
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub
    	
    	/*
    	List<String> sensor_features = Arrays.asList("angle", "curLapTime", "damage","distFromStart","distRaced","fuel","gear","lastLapTime",
    			"opponents","racePos","rpm","speedX","speedY","speedZ","track","trackPos","wheelSpinVel","z","focus");
    	
    	List<String> action_features = Arrays.asList("accel","brake","clutch","gear","steer","meta","focus");
		*/
    	
    	List<String> sensorFeatures = Arrays.asList("angle", "speedX", "speedY", "track", "trackPos");
    	
    	List<String> actionFeatures = Arrays.asList("steer");
    	
    	String logPath = "logs/humanlog_1447938562784.txt";
    	
    	int maxLaps = 20;
    	
    	train(logPath,sensorFeatures,actionFeatures, maxLaps);

	}
	
	public static void train(String logPath, List<String> sensorFeatures, List<String> actionFeatures, int maxLaps){
		
		System.out.println("Reading log file: "+logPath);

    	double[][] x = new double[1][1];
    	double[][] d = new double[1][1];
		
		int i = 0; //nr of samples
		int lineNr = 1;
		double lastDistance = 0;
		int lap = 0;
		double[] sums = new double[sensorFeatures.size()];
    	
		try(BufferedReader br = new BufferedReader(new FileReader(logPath))) {
    	    for(String line; (line = br.readLine()) != null; ) {
    	    	
    	    	if((lineNr)%1000==0) System.out.println(lineNr+" lines processed");
    	    		
    	        // process the line.
    	    	//System.out.println(line);
    	    	
    	    	//List<Double> feats = new ArrayList<Double>();
	    		//Double[] vals_doubles;
    	    	if (Pattern.compile("^Received: \\(.*").matcher(line).matches()){
    	    		Matcher m = Pattern.compile("\\("+"distFromStart"+"\\s(-?\\d*(\\.)?\\d*(E-|e-)?\\d*+(\\s)?)*\\)").matcher(line);
    	    		if (m.find()){
		    	    	double currentDistance = Double.parseDouble(m.group(0).replaceAll("\\("+"distFromStart"+"\\s((-?\\d*(\\.)?\\d*(E-|e-)?\\d*+(\\s)?)*)\\)", "$1"));
		    	    	if (lastDistance - currentDistance > 1000)
		    	    		lap ++;
		    	    	
		    	    	if (lap > maxLaps)
		    	    		break;
		    	    	lastDistance = currentDistance;
    	    		}
    	    		
		    		int j = 0;
	    	    	for(String sensor_feat : sensorFeatures){
		    	    	
		    	    	m = Pattern.compile("\\("+sensor_feat+"\\s(-?\\d*(\\.)?\\d*(E-|e-)?\\d*+(\\s)?)*\\)").matcher(line);
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
		    	    			sums[j] += Double.parseDouble(v);
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
	    	    	for(String action_feat : actionFeatures){
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
    	    	lineNr ++;
    	    }
    	}catch(IOException e){
    		System.out.println("ERROR while opening file");
    		e.printStackTrace();
    	}
		
		// normalize data (see var sums) sums /= i;
		for (int z=0;z<sums.length;z++)
			sums[z] = sums[z]/i;

    	MLDataSet trainingSet = new BasicMLDataSet(x, d);
    	
    	int nHidden = 100;
    	double thresholdError = 0.001;
    	int maxEpochs = 200;
    	
    	BasicNetwork net = new BasicNetwork();
    	net.addLayer(new BasicLayer(null, true, sensorFeatures.size()));
    	net.addLayer(new BasicLayer(new ActivationTANH(), true, nHidden));
    	net.addLayer(new BasicLayer(new ActivationTANH(), true, nHidden));
    	net.addLayer(new BasicLayer(new ActivationLinear(), false, actionFeatures.size()));
    	
    	net.getStructure().finalizeStructure();
    	net.reset();
    	
    	ResilientPropagation train = new ResilientPropagation(net, trainingSet);
    	int epoch = 1 ;
    	do {
    		train.iteration();
    		System.out.println("Epoch: "+epoch+" Error: "+train.getError());
    		epoch++;
    	}while(train.getError()>thresholdError && epoch<maxEpochs);
    	train.finishTraining();
    	
    	// make a prediction
    	System.out.println("true: "+d[0][0]);
    	System.out.println("pred: "+net.compute(trainingSet.get(0).getInput()));
    	
    	System.out.println("Trained weights: "+net.dumpWeights());
    	
    	saveNetwork(net);
    	
    	return;
	}
	
	private static void saveNetwork(BasicNetwork net){
		System.out.println("Saving network");
		try {
			SerializeObject.save(new File(outputFileName), net);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		System.out.println("Saving network finished");
		
		return;
	}
	
	public static BasicNetwork loadNetwork(String filename){
		BasicNetwork network = null;
		
		System.out.println("Loading network");
		
		try {
			network = (BasicNetwork) SerializeObject.load(new File(filename));
		} catch (ClassNotFoundException | IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return network;
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

}
