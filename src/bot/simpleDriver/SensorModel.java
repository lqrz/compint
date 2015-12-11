package bot.simpleDriver;

//
//Source code recreated from a .class file by IntelliJ IDEA
//(powered by Fernflower decompiler)
//

public interface SensorModel {
 double getSpeed();

 double getAngleToTrackAxis();

 double[] getTrackEdgeSensors();

 double[] getFocusSensors();

 double getTrackPosition();

 int getGear();

 double[] getOpponentSensors();

 int getRacePosition();

 double getLateralSpeed();

 double getCurrentLapTime();

 double getDamage();

 double getDistanceFromStartLine();

 double getDistanceRaced();

 double getFuelLevel();

 double getLastLapTime();

 double getRPM();

 double[] getWheelSpinVelocity();

 double getZSpeed();

 double getZ();

 String getMessage();

 float[] getInitialAngles();

 void setInitialAngles(float[] var1);
}
