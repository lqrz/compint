package bot.simpleExample2;

import cicontest.algorithm.abstracts.AbstractDriver;
import cicontest.algorithm.abstracts.DriversUtils;
import cicontest.torcs.client.Action;
import cicontest.torcs.client.SensorModel;
import cicontest.torcs.genome.IGenome;
import cicontest.torcs.controller.extras.ABS;
import cicontest.torcs.controller.extras.AutomatedClutch;
import cicontest.torcs.controller.extras.AutomatedGearbox;
import cicontest.torcs.controller.extras.AutomatedRecovering;

public class DefaultDriver extends AbstractDriver {

    NeuralNetwork neuralNetwork = new NeuralNetwork();

    public DefaultDriver() {
        initialize();
        //neuralNetwork = neuralNetwork.loadGenome();
    }

    public void loadGenome(IGenome genome) { }

    public void initialize(){
       this.enableExtras(new AutomatedClutch());
       this.enableExtras(new AutomatedGearbox());
       this.enableExtras(new AutomatedRecovering());
       this.enableExtras(new ABS());
    }

    @Override
    public void control(Action action, SensorModel sensors) {
        // Example of a bot that drives pretty well; you can use this to generate data
        action.steering = DriversUtils.alignToTrackAxis(sensors, 0.5);
        if(sensors.getSpeed() > 60.0D) { //60
            action.accelerate = 0.0D;
            action.brake = 0.0D;
        }

        if(sensors.getSpeed() > 70.0D) { //70
            action.accelerate = 0.0D;
            action.brake = -1.0D;
        }

        if(sensors.getSpeed() <= 60.0D) { //60
            action.accelerate = (80.0D - sensors.getSpeed()) / 80.0D;
            action.brake = 0.0D;
        }

        if(sensors.getSpeed() < 30.0D) { //30
            action.accelerate = 1.0D;
            action.brake = 0.0D;
        }

//        System.out.println(action.steering +"steering");
//        System.out.println(action.accelerate + "acceleration");
//        System.out.println(action.brake + "brake");
    }

    public String getDriverName() {
        return "simple example 2";
    }

    public void controlQualification(Action action, SensorModel sensors) { }

    public void controlRace(Action action, SensorModel sensors) {}

    public void defaultControl(Action action, SensorModel sensors){}

    @Override
    public double getSteering(SensorModel sensorModel) {
        return 0;
    }

    @Override
    public double getAcceleration(SensorModel sensorModel) {
        return 0;
    }

    public double getBraking(SensorModel sensorModel){
        return 0;
    }

}