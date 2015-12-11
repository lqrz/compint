package team33.humanDriver;

import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.io.FileNotFoundException;
import java.io.PrintWriter;

import cicontest.algorithm.abstracts.AbstractDriver;
import cicontest.torcs.client.Action;
import cicontest.torcs.client.SensorModel;
import cicontest.torcs.genome.IGenome;

public class HumanDriver extends AbstractDriver implements KeyListener {

    private PrintWriter out;
    private IsKeyPressed keylistener = new IsKeyPressed();
    private String key;
    private int key_int;

	public void loadGenome(IGenome genome) {
        if (genome instanceof HumanDriverGenome) {
            HumanDriverGenome MyGenome = (HumanDriverGenome) genome;
        } else {
            System.err.println("Invalid Genome assigned");
        }
        java.util.Date date= new java.util.Date();
        System.out.println("logging to logs/humanlog_"+ date.getTime() +".txt");
        try {
			this.out = new PrintWriter("logs/humanlog_"+ date.getTime() +".txt");
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }

    public double getAcceleration(SensorModel sensors) {
    	if (this.keylistener.isUpPressed())
    		return this.keylistener.upAmount();
    	else
    		return 0;
    }
    
    public double getBrake(SensorModel sensors) {
    	if (this.keylistener.isDownPressed())
    		return 1;
    	else
    		return 0;
    }

    public double getSteering(SensorModel sensors){
    	if (this.keylistener.isLeftPressed())
    		return -1 * this.keylistener.leftAmount();
    	else if (this.keylistener.isRightPressed())
    		return this.keylistener.rightAmount();
    	else
    		return 0;
    }
    
    public String getDriverName() {
        return "simple example";
    }

    public void controlQualification(Action action, SensorModel sensors) {;
    	System.out.println("Control qualification");
    
        action.clutch = 1;
        action.steering =  Math.random() * (1 - -1)  -1;
        action.accelerate = 1;
        action.brake = 0;
        //super.controlQualification(action, sensors)
    }

    public void controlRace(Action action, SensorModel sensors) {
//    	System.out.println("calling controlRace");
        this.out.println("Received: " + sensors.getMessage());
        action.clutch = 1;
        action.steering = this.getSteering(sensors);
        action.accelerate = this.getAcceleration(sensors);
        action.brake = this.getBrake(sensors);
//        System.out.printf("Steering: %f, Acceleration: %f, Brake:%f\n", action.steering, action.accelerate, action.brake);
        this.out.println("Sending: " + action.toString());
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

	@Override
	public void keyPressed(KeyEvent e) {
		// TODO Auto-generated method stub
		System.out.println("Pressed key: "+KeyEvent.getKeyText(e.getKeyCode()));
		this.key_int = e.getKeyCode();
		this.key = KeyEvent.getKeyText(this.key_int);
	}

	@Override
	public void keyReleased(KeyEvent e) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void keyTyped(KeyEvent e) {
		// TODO Auto-generated method stub
		
	}
    
}