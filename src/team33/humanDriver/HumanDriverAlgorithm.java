package team33.humanDriver;

import java.io.File;
import cicontest.algorithm.abstracts.AbstractAlgorithm;
import cicontest.algorithm.abstracts.AbstractRace;
import cicontest.algorithm.abstracts.DriversUtils;
import cicontest.torcs.controller.Driver;
import cicontest.torcs.controller.Human;
import race.TorcsConfiguration;

public class HumanDriverAlgorithm extends AbstractAlgorithm {

    private static final long serialVersionUID = 654963126352653L;

    HumanDriverGenome[] drivers = new HumanDriverGenome[1];
    int [] results = new int[1];

    public Class<? extends Driver> getDriverClass(){
        return HumanDriver.class;
    }

    public void run(boolean continue_from_checkpoint) {
        if(!continue_from_checkpoint){
            HumanDriverGenome genome = new  HumanDriverGenome();
            drivers[0] = genome;

            //Start a race
            HumanRace race = new HumanRace();
            race.setTrack( AbstractRace.DefaultTracks.getTrack(0));
            race.laps = 1;

            //for speedup set withGUI to false
            results = race.runRace(drivers, true);

            // Save genome/nn
            DriversUtils.storeGenome(drivers[0]);
        }
            // create a checkpoint this allows you to continue this run later
            DriversUtils.createCheckpoint(this);
            //DriversUtils.clearCheckpoint();
    }

    public static void main(String[] args) {

        //Set path to torcs.properties
    	TorcsConfiguration.getInstance().initialize(new File("torcs.properties"));
		/*
		 *
		 * Start without arguments to run the algorithm
		 * Start with -continue to continue a previous run
		 * Start with -show to show the best found
		 * Start with -show-race to show a race with 10 copies of the best found
		 * Start with -human to race against the best found
		 *
		 */
        HumanDriverAlgorithm algorithm = new HumanDriverAlgorithm();
        DriversUtils.registerMemory(algorithm.getDriverClass());
        if(args.length > 0 && args[0].equals("-show")){
            new HumanRace().showBest();
        } else if(args.length > 0 && args[0].equals("-show-race")){
            new HumanRace().showBestRace();
        } else if(args.length > 0 && args[0].equals("-human")){
            new HumanRace().raceBest();
        } else if(args.length > 0 && args[0].equals("-continue")){
            if(DriversUtils.hasCheckpoint()){
                DriversUtils.loadCheckpoint().run(true);
            } else {
                algorithm.run();
            }
        } else {
            algorithm.run();
        }
    }

}