package old.team33.example;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import old.team33.NeuralNetwork;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.neural.networks.BasicNetwork;

import team33.InputReader;
import team33.NNPredictor;
import team33.activationFunctions.Sigmoid;
import team33.activationFunctions.Tanh;
import cicontest.algorithm.abstracts.AbstractDriver;
import cicontest.torcs.client.Action;
import cicontest.torcs.client.SensorModel;
import cicontest.torcs.genome.IGenome;

public class DefaultDriver extends AbstractDriver {

    private NNPredictor nn_steering;
    private NNPredictor nn_accel;
    private List<String> inputfields;
    private boolean init = true;

    public void loadGenome(IGenome genome) {
        if (genome instanceof DefaultDriverGenome) {
            DefaultDriverGenome MyGenome = (DefaultDriverGenome) genome;
            this.nn_steering = MyGenome.getNNsteer();
            this.nn_accel = MyGenome.getNNaccel();
            this.inputfields = Arrays.asList(MyGenome.getInputFields());
        } else {
            System.err.println("Invalid Genome assigned");
        }
    }

    public double getAcceleration(SensorModel sensors) {
        double[] sensorArray = new double[4];
//        double output = this.nn.getOutput(sensors);
    return 1;
    }

    public double getSteering(SensorModel sensors){
		return 0;
//        Double output = this.nn.getOutput(sensors);
//        return 0.5;
//    	double[] output = this.nn.getSteering(sensors);
//    	System.out.println("NN computed steering: "+output);
//    	
//    	return output;
    }

    public String getDriverName() {
        return "Team33";
    }

    public void controlQualification(Action action, SensorModel sensors) {;
    	System.out.println("Control qualification");
    
        action.clutch = 1;
        action.steering =  Math.random() * (1 - -1)  -1;
        action.accelerate = 1;
        action.brake = 0;
        //super.controlQualification(action, sensors)
    }
    
    private void AddInputs(ArrayList<Double> x, SensorModel sensors) {
    	if(this.inputfields.contains("angle")) {
    		x.add(sensors.getAngleToTrackAxis());
    	} else if(this.inputfields.contains("curLapTime")) {
    		x.add(sensors.getCurrentLapTime());
    	} else if(this.inputfields.contains("damage")) {
    		x.add(sensors.getDamage());
    	} else if(this.inputfields.contains("distFromStart")) {
    		x.add(sensors.getDistanceFromStartLine());
    	} else if(this.inputfields.contains("distRaced")) {
    		x.add(sensors.getDistanceRaced());
    	} else if(this.inputfields.contains("focus")) {
    		x.addAll(Arrays.asList(ArrayUtils.toObject(sensors.getFocusSensors())));
    	} else if(this.inputfields.contains("fuel")) {
    		x.add(sensors.getFuelLevel());
    	} else if(this.inputfields.contains("gear")) {
    		x.add((double)sensors.getGear());
    	} else if(this.inputfields.contains("lastLapTime")) {
    		x.add(sensors.getLastLapTime());
    	} else if(this.inputfields.contains("opponents")) {
    		x.addAll(Arrays.asList(ArrayUtils.toObject(sensors.getOpponentSensors())));
    	} else if(this.inputfields.contains("racePos")) {
    		x.add((double)sensors.getRacePosition());
    	} else if(this.inputfields.contains("rpm")) {
    		x.add(sensors.getRPM());
    	} else if(this.inputfields.contains("speedX")) {
    		x.add(sensors.getSpeed());
    	} else if(this.inputfields.contains("speedY")) {
    		x.add(sensors.getLateralSpeed());
    	} else if(this.inputfields.contains("speedZ")) {
    		x.add(sensors.getZSpeed());
    	} else if(this.inputfields.contains("track")) {
    		x.addAll(Arrays.asList(ArrayUtils.toObject(sensors.getTrackEdgeSensors())));
    	} else if(this.inputfields.contains("trackPos")) {
    		x.add(sensors.getTrackPosition());
    	} else if(this.inputfields.contains("wheelSpinVel")) {
    		x.addAll(Arrays.asList(ArrayUtils.toObject(sensors.getWheelSpinVelocity())));
    	} else if(this.inputfields.contains("z")) {
    		x.add(sensors.getZ());
    	}
    }

    public void controlRace(Action action, SensorModel sensors) {
//        System.out.println("Control race");
        
//    	System.out.println(sensors.getGear());
        action.clutch = 1;
//        action.steering =  Math.random() * (1 - -1)  -1;

//        double[] a = this.nn.getSteering(sensors);
        // from training: sensorFeatures = ("angle", "speedX", "speedY", "track", "trackPos")
        
		ArrayList<Double> xarray = new ArrayList<>();
		this.AddInputs(xarray, sensors);
		double[] input = ArrayUtils.toPrimitive(xarray.toArray(new Double[xarray.size()]));
		
		RealMatrix input_column_matrix = new Array2DRowRealMatrix(input);
		
		double steering = this.nn_steering.predict(input_column_matrix, new Sigmoid(), new Tanh());
		double accel = this.nn_accel.predict(input_column_matrix, new Sigmoid(), new Tanh());
		
//        double steering = this.nn_steering.compute(new BasicMLData(a)).getData(0);
//        double accel = this.nn_accel.compute(new BasicMLData(a)).getData(0);
        
//        System.out.println("Predicted steering: "+steering);

//        action.accelerate = a[0];
        action.accelerate = accel;
        
        action.steering = steering;

//        if (a[0] > 0){
//        	action.accelerate = a[0];
//        }else{
//        	action.accelerate = 0.5;
//        }
        
//        System.out.println("NN steering: "+action.steering+" NN acceleration: "+action.accelerate);
        System.out.println("Sending: (steer "+action.steering+")(accel "+action.accelerate+")");
        action.brake = 0;
        
//        if (this.init){
//        	action.steering = 0;
//        	action.accelerate = 1;
//        	this.init = false;
//        }
        
        //super.ControlRace(action, sensors);
    }

    public void defaultControl(Action action, SensorModel sensors){
        System.out.println("Default control");
        action.clutch = 1;
        action.steering =  Math.random() * (1 - -1)  -1;
        action.accelerate = 1;
        action.brake = 0;
        //super.defaultControl(action, sensors);
    }
}