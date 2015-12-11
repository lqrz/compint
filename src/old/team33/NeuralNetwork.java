package old.team33;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;

import old.team33.ESN;
import old.team33.Utils;

import org.apache.commons.math3.analysis.function.Tanh;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.lang3.ArrayUtils;

import cicontest.torcs.client.SensorModel;
import net.razorvine.pickle.PickleException;


public class NeuralNetwork {

	private ESN nn;
	private Utils utils;
	private RealVector steering_xn = null;
	private RealVector steering_yn = MatrixUtils.createRealVector(new double[1]);
	private RealMatrix w_in_real;
	private RealMatrix w_real;
	private RealMatrix w_out_real;
	private RealMatrix w_back_real;
//	private RealVector x_n_real;
	private boolean concatenated;
    private PrintWriter out;
    private double bias_value = 1;

	public NeuralNetwork() {
		super();
		// TODO Auto-generated constructor stub
	}

	public NeuralNetwork(boolean concatenated){
		this.nn = new ESN();
		this.utils = new Utils();
		this.initialize();
		this.concatenated = concatenated;
	}

	public void initialize(){
		ArrayList<ArrayList<Double>> w_out = null;
		ArrayList<ArrayList<Double>> w_in = null;
		ArrayList<ArrayList<Double>> w = null;
		ArrayList<ArrayList<Double>> w_back = null;
		ArrayList<Double> x_n = null;
		
		System.out.println("Loading pickle files");
		
		try {
			w_out = this.nn.loadPickleMatrix("trained_files\\w_out.p");
			w_in = this.nn.loadPickleMatrix("trained_files\\w_in.p");
			w = this.nn.loadPickleMatrix("trained_files\\w.p");
			w_back = this.nn.loadPickleMatrix("trained_files\\w_back.p");
			x_n= this.nn.loadPickleVector("trained_files\\x_n.p");
		} catch (PickleException | IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		System.out.println("Converting pickle structures to primitive arrays");

		double[][] w_in_prim = this.utils.toPrimitive(w_in);
		double[][] w_prim = this.utils.toPrimitive(w);
		double[][] w_out_prim = this.utils.toPrimitive(w_out);
		double[][] w_back_prim = this.utils.toPrimitive(w_back);
		double[] x_n_prim = this.utils.toPrimitiveVector(x_n);
		
		System.out.println("Converting primitive structures to Real structures");
		
		this.w_in_real = MatrixUtils.createRealMatrix(w_in_prim);
		this.w_real = MatrixUtils.createRealMatrix(w_prim);
		this.w_out_real = MatrixUtils.createRealMatrix(w_out_prim);
		this.w_back_real = MatrixUtils.createRealMatrix(w_back_prim);
//		this.x_n_real = MatrixUtils.createRealVector(x_n_prim);
		
        try {
			this.out = new PrintWriter("logs/humanlog.txt");
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		System.out.println("finished init");
	}

//	NeuralNetwork(int inputs, int hidden, int outputs){}

	public double getOutput(SensorModel sensor_data) {
		return 0.5;
	}
	
	public double[] getSteering(SensorModel sensor_data){
		
		this.out.println("Received: "+sensor_data.getMessage());
		System.out.println(sensor_data.getMessage());

		
		if(this.steering_xn == null){
			this.steering_xn = MatrixUtils.createRealVector(new double[w_real.getColumnDimension()]);
//			this.steering_xn = x_n_real;
		}
		
//		Get same features as used for training
		double[] a = new double[]{sensor_data.getAngleToTrackAxis()};
		//a = ArrayUtils.add(a, sensor_data.getCurrentLapTime());
		//a = ArrayUtils.add(a, sensor_data.getDamage());
		//a = ArrayUtils.add(a, sensor_data.getDistanceFromStartLine());
		//a = ArrayUtils.add(a, sensor_data.getDistanceRaced());
		//a = ArrayUtils.add(a, sensor_data.getFuelLevel());
		
		//a = ArrayUtils.add(a,(double) sensor_data.getGear());
		
		//a = ArrayUtils.add(a, sensor_data.getLastLapTime());
		//a = ArrayUtils.addAll(a, sensor_data.getOpponentSensors());
		//a = ArrayUtils.add(a, (double) sensor_data.getRacePosition());
		
//		a = ArrayUtils.add(a, sensor_data.getRPM());
//		a = ArrayUtils.add(a, sensor_data.getSpeed()); //speedX
//		a = ArrayUtils.add(a, sensor_data.getLateralSpeed()); //speedY
//		a = ArrayUtils.add(a, sensor_data.getZSpeed()); //speedZ
		
		a = ArrayUtils.addAll(a, sensor_data.getTrackEdgeSensors());
		
//		a = ArrayUtils.add(a, sensor_data.getTrackPosition());
		
		//a = ArrayUtils.addAll(a, sensor_data.getWheelSpinVelocity());
		
		a = ArrayUtils.add(a, sensor_data.getZ()); //z
		
		a = ArrayUtils.addAll(a, sensor_data.getFocusSensors()); //focus
		
//		System.out.println("sensor data: "+ArrayUtils.toString(a));
//		System.out.println(MatrixUtils.createRealVector(a).getDimension());
		
		RealVector uno = w_in_real.operate(MatrixUtils.createRealVector(new double[]{this.bias_value}).append(MatrixUtils.createRealVector(a)));
		//TODO: whats the first x_n for prediction?
		RealVector dos = w_real.operate(this.steering_xn);
		//System.out.println(w_out_real.transpose().getRowDimension()+"x"+w_out_real.transpose().getColumnDimension());
		//System.out.println(dos.getDimension());
//		RealVector tres = w_out_real.transpose().operate(dos);
		RealVector tres = w_back_real.operate(this.steering_yn);
		this.steering_xn = ((uno.add(dos)).add(tres)).map(new Tanh());
		
		//System.out.println(w_back_real.getRowDimension()+"x"+w_back_real.getColumnDimension());
//		this.steering_xn = w_back_real.operate(tres);
		
		//System.out.println(tres.toString());
		if (this.concatenated){
			this.steering_yn = ((w_out_real.transpose()).operate(this.steering_xn.append(MatrixUtils.createRealVector(a)))).map(new Tanh());
		}else{
			this.steering_yn = w_out_real.transpose().operate(this.steering_xn).map(new Tanh());
		}
		double[] ret = new double[2];
		ret[0] = this.steering_yn.getEntry(0);
//		ret[1] = this.steering_yn.getEntry(1);
		return ret;
		}
	
	//TODO: remove this override
	public double getSteering(double[] a){
		/*
		 * This method is for testing. Used in file NeuralNetworkOutput.java
		 */
		ArrayList<ArrayList<Double>> w_out = null;
		ArrayList<ArrayList<Double>> w_in = null;
		ArrayList<ArrayList<Double>> w = null;
		ArrayList<ArrayList<Double>> w_back = null;
		ArrayList<Double> x_n = null;
		try {
			w_out = this.nn.loadPickleMatrix("trained_files\\w_out.p");
			w_in = this.nn.loadPickleMatrix("trained_files\\w_in.p");
			w = this.nn.loadPickleMatrix("trained_files\\w.p");
			w_back = this.nn.loadPickleMatrix("trained_files\\w_back.p");
			x_n= this.nn.loadPickleVector("trained_files\\x_n.p");
		} catch (PickleException | IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		double[][] w_in_prim = this.utils.toPrimitive(w_in);
		double[][] w_prim = this.utils.toPrimitive(w);
		double[][] w_out_prim = this.utils.toPrimitive(w_out);
		double[][] w_back_prim = this.utils.toPrimitive(w_back);
		double[] x_n_prim = this.utils.toPrimitiveVector(x_n);
		

		RealMatrix w_in_real = MatrixUtils.createRealMatrix(w_in_prim);
		RealMatrix w_real = MatrixUtils.createRealMatrix(w_prim);
		RealMatrix w_out_real = MatrixUtils.createRealMatrix(w_out_prim);
		RealMatrix w_back_real = MatrixUtils.createRealMatrix(w_back_prim);
		RealVector x_n_real = MatrixUtils.createRealVector(x_n_prim);
		
		if(this.steering_xn == null){
			this.steering_xn = x_n_real;
		}
		
		RealVector uno = w_in_real.operate(MatrixUtils.createRealVector(a));
		//TODO: whats the first x_n for prediction?
		RealVector dos = w_real.operate(this.steering_xn);
		//System.out.println(w_out_real.transpose().getRowDimension()+"x"+w_out_real.transpose().getColumnDimension());
		//System.out.println(dos.getDimension());
//		RealVector tres = w_out_real.transpose().operate(dos);
		RealVector tres = w_back_real.operate(this.steering_yn);
		System.out.println(w_back_real.getRowDimension()+"x"+w_back_real.getColumnDimension());
		System.out.println(this.steering_yn.getDimension());
		this.steering_xn = uno.add(dos).add(tres).map(new Tanh());
		
		//System.out.println(w_back_real.getRowDimension()+"x"+w_back_real.getColumnDimension());
//		this.steering_xn = w_back_real.operate(tres);
		
		//System.out.println(tres.toString());
		
		if (this.concatenated){
			this.steering_yn = w_out_real.transpose().operate(this.steering_xn.append(MatrixUtils.createRealVector(a))).map(new Tanh());
		}else{
			this.steering_yn = w_out_real.transpose().operate(this.steering_xn).map(new Tanh());
		}
		
//		return this.steering_yn.getEntry(0);
		System.out.println(this.steering_yn.getEntry(0));
		System.out.println(this.steering_yn.getEntry(1));
		return this.steering_yn.getEntry(0);
	}
}
