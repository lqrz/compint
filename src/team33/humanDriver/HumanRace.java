package team33.humanDriver;

import cicontest.algorithm.abstracts.AbstractRace;
import cicontest.algorithm.abstracts.DriversUtils;
import cicontest.torcs.controller.Driver;
import cicontest.torcs.controller.Human;

public class HumanRace extends AbstractRace {

	public int[] runQualification(HumanDriverGenome[] drivers, boolean withGUI){
		HumanDriver[] driversList = new HumanDriver[drivers.length + 1 ];
		for(int i=0; i<drivers.length; i++){
			driversList[i] = new HumanDriver();
			driversList[i].loadGenome(drivers[i]);
		}
		return runQualification(driversList, withGUI);
	}

	
	public int[] runRace(HumanDriverGenome[] drivers, boolean withGUI){
		int size = Math.min(10, drivers.length);
		HumanDriver[] driversList = new HumanDriver[size];
		for(int i=0; i<size; i++){
			driversList[i] = new HumanDriver();
			driversList[i].loadGenome(drivers[i]);
		}
		return runRace(driversList, withGUI, true);
	}

	
	
	public void showBest(){
		if(DriversUtils.getStoredGenome() == null ){
			System.err.println("No best-genome known");
			return;
		}
		
		HumanDriverGenome best = (HumanDriverGenome) DriversUtils.getStoredGenome();
		HumanDriver driver = new HumanDriver();
		driver.loadGenome(best);
		
		HumanDriver[] driversList = new HumanDriver[1];
		driversList[0] = driver;
		runQualification(driversList, true);
	}
	
	public void showBestRace(){
		if(DriversUtils.getStoredGenome() == null ){
			System.err.println("No best-genome known");
			return;
		}
	
		HumanDriver[] driversList = new HumanDriver[1];
		
		for(int i=0; i<10; i++){
			HumanDriverGenome best = (HumanDriverGenome) DriversUtils.getStoredGenome();
			HumanDriver driver = new HumanDriver();
			driver.loadGenome(best);
			driversList[i] = driver;
		}
		
		runRace(driversList, true, true);
	}
	
	public void raceBest(){
		
		if(DriversUtils.getStoredGenome() == null ){
			System.err.println("No best-genome known");
			return;
		}
		
		Driver[] driversList = new Driver[10];
		for(int i=0; i<10; i++){
			HumanDriverGenome best = (HumanDriverGenome) DriversUtils.getStoredGenome();
			HumanDriver driver = new HumanDriver();
			driver.loadGenome(best);
			driversList[i] = driver;
		}
		driversList[0] = new Human();
		runRace(driversList, true, true);
	}
}
