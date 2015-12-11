package old.team33;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;

public class EvoNetwork {
	private static final String[] TORCS_ARGS = new String[] {"-T", "-nofuel", "-nodamage", "-nolaptime", "-noisy"};
	// TODO population of networks
	
	// TODO drive function
	
	// TODO evolution function
	
	// TODO save best network function


	public static void main(String[] args) {
		if (args.length == 0) {
			System.out.println("Usage: java EvoNetwork.java <path/to/wtorcs.exe>");
			return;
		}
		File torcs_path = new File(args[0]);
		ArrayList<String> cmd = new ArrayList<String>(Arrays.asList(torcs_path.getAbsolutePath()));
		for (String arg : TORCS_ARGS) {
			cmd.add(arg);
		}
		System.out.println("starting " + cmd.toString());
		ProcessBuilder pb = new ProcessBuilder(cmd);
		pb.directory(torcs_path.getParentFile());
		
		// TODO generate random initial population of echo state networks
		
		// TODO evolution loop: take network, start client, start torcs, get results
		
		try {
			Process torcs = pb.start();
			InputStream is = torcs.getInputStream();
	        InputStreamReader isr = new InputStreamReader(is);
	        BufferedReader br = new BufferedReader(isr);
	        String line;
	        System.out.println("TORCS output:");
	        while ((line = br.readLine()) != null) {
	            System.out.println(line);
	        }
	        System.out.println("TORCS EXIT");
	        // TODO get time, distance from start, lap count, finished track
		} catch (IOException e) {
			System.out.println("ERROR starting TORCS game:");
			e.printStackTrace();
		}

		/*
    	List<String> sensor_features = Arrays.asList("angle", "curLapTime", "damage","distFromStart","distRaced","fuel","gear","lastLapTime",
    			"opponents","racePos","rpm","speedX","speedY","speedZ","track","trackPos","wheelSpinVel","z","focus");

    	List<String> action_features = Arrays.asList("accel","brake","clutch","gear","steer","meta","focus");
		 */

		//    	List<String> sensorFeatures = Arrays.asList("angle", "speedX", "speedY", "track", "trackPos");
		//    	
		//    	List<String> actionFeatures = Arrays.asList("steer");
		//    	
		//    	String logPath = "logs/humanlog_1447938562784.txt";
		//    	
		//    	int maxLaps = 20;
		//    	
		//    	train(logPath,sensorFeatures,actionFeatures, maxLaps);

	}
}