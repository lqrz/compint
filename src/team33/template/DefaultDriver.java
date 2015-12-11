package team33.template;

import cicontest.algorithm.abstracts.AbstractDriver;
import cicontest.torcs.client.Action;
import cicontest.torcs.client.SensorModel;
import cicontest.torcs.genome.IGenome;
import team33.NNPredictor;
import cicontest.torcs.controller.extras.ABS;
import cicontest.torcs.controller.extras.AutomatedClutch;
import cicontest.torcs.controller.extras.AutomatedGearbox;
import cicontest.torcs.controller.extras.AutomatedRecovering;

public class DefaultDriver extends AbstractDriver {
	
	private NNPredictor nn_brake;
	private NNPredictor nn_accel;
	private NNPredictor nn_steer;
	
	public DefaultDriver(){
		this.initialize();
	}
	
	private void initialize(){
		this.enableExtras(new AutomatedClutch());
		this.enableExtras(new AutomatedGearbox());
		this.enableExtras(new AutomatedRecovering());
		this.enableExtras(new ABS());
		
		nn_brake = new NNPredictor("D:/Dropbox/Java/Computational Intelligence/trained_files/mlp/SimpleDriver/brake1449536327.67.txt");
		nn_accel = new NNPredictor("D:/Dropbox/Java/Computational Intelligence/trained_files/mlp/SimpleDriver/acceleration1449536327.67.txt");
		nn_steer = new NNPredictor("D:/Dropbox/Java/Computational Intelligence/trained_files/mlp/SimpleDriver/steering1449536327.67.txt");
	}

	@Override
	public void control(Action action, SensorModel sensors) {
		action.steering = this.getSteering(sensors);
		action.accelerate = this.getAcceleration(sensors);
		action.brake = this.getBraking(sensors);
	}

    public void loadGenome(IGenome genome) {
//        if (genome instanceof DefaultDriverGenome) {
//            DefaultDriverGenome MyGenome = (DefaultDriverGenome) genome;
//        } else {
//            System.err.println("Invalid Genome assigned");
//        }
    }

    public double getAcceleration(SensorModel sensors) {
        return this.nn_accel.predict(sensors);
    }

    public double getSteering(SensorModel sensors){
        return this.nn_steer.predict(sensors);
    }

    public double getBraking(SensorModel sensors) {
        return this.nn_brake.predict(sensors);
    }

    public String getDriverName() {
        return "Team33";
    }

    /*
     * The following methods are only here as a reminder that you can,
     * and may change all driver methods, including the already implemented
     * ones, such as those beneath.
     */

    public void controlQualification(Action action, SensorModel sensors) {
           super.controlQualification(action, sensors);
    }

    public void controlRace(Action action, SensorModel sensors) {
            super.controlRace(action, sensors);
    }

    public void defaultControl(Action action, SensorModel sensors){
            super.defaultControl(action, sensors);
    }
}