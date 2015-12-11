package team33;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

//import org.apache.commons.math3.analysis.function.Tanh;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.MathUtils;
import org.apache.commons.lang3.ArrayUtils;

import team33.activationFunctions.Activation;
import team33.activationFunctions.Sigmoid;
import team33.activationFunctions.Tanh;
import team33.activationFunctions.ReLU;
import team33.activationFunctions.Linear;

import cicontest.torcs.client.SensorModel;

public class NNPredictor {
	private static final String WhitespaceRegex = "[\t ]"; 
	private static final Map<String, Activation> ActivationMap;
	static {
		ActivationMap = new HashMap<>();
		ActivationMap.put("sigmoid", new Sigmoid());
		ActivationMap.put("tanh", new Tanh());
		ActivationMap.put("relu", new ReLU());
		ActivationMap.put("linear", new Linear());
	}
	private RealMatrix w_hidden_in;
	private RealMatrix w_out_hidden;
	private String[] fields;
	private RealMatrix mean;
	private RealMatrix std;
	private ArrayList<Activation> actv_funcs;
	
	public NNPredictor() {
		super();
		this.clear();
	}
	
	public NNPredictor(String filename) {
		this();
		ReadFile(filename);
	}
	
	private void clear() {
		this.fields = null;
		this.w_hidden_in = null;
		this.w_out_hidden = null;
		this.mean = null;
		this.std = null;
		this.actv_funcs = new ArrayList<>();
	}
	
	public RealMatrix getMean() {
		return this.mean;
	}
	
	public RealMatrix getStd() {
		return this.std;
	}
	
	private RealMatrix AppendOne(RealMatrix x) {
		double[][] data = x.getData();
		RealMatrix result = x.createMatrix(x.getRowDimension() + 1, x.getColumnDimension());
		for (int i = 0; i < data.length; i++)
			result.setEntry(i, 0, data[i][0]);
		result.setEntry(result.getRowDimension() - 1, 0, 1.0);
		
		return result;
	}
	
	private void AddInputs(ArrayList<Double> x, SensorModel sensors) {
		for(String field : this.fields) {
			if (field.equals("angle")) {
				x.add(sensors.getAngleToTrackAxis());
			} else if (field.equals("curLapTime")) {
				x.add(sensors.getCurrentLapTime());
			} else if (field.equals("damage")) {
				x.add(sensors.getDamage());
			} else if (field.equals("distFromStart")) {
				x.add(sensors.getDistanceFromStartLine());
			} else if (field.equals("distRaced")) {
				x.add(sensors.getDistanceRaced());
			} else if (field.equals("focus")) {
				x.addAll(Arrays.asList(ArrayUtils.toObject(sensors.getFocusSensors())));
			} else if (field.equals("fuel")) {
				x.add(sensors.getFuelLevel());
			} else if (field.equals("gear")) {
				x.add((double) sensors.getGear());
			} else if (field.equals("lastLapTime")) {
				x.add(sensors.getLastLapTime());
			} else if (field.equals("opponents")) {
				x.addAll(Arrays.asList(ArrayUtils.toObject(sensors.getOpponentSensors())));
			} else if (field.equals("racePos")) {
				x.add((double) sensors.getRacePosition());
			} else if (field.equals("rpm")) {
				x.add(sensors.getRPM());
			} else if (field.equals("speedX")) {
				x.add(sensors.getSpeed());
			} else if (field.equals("speedY")) {
				x.add(sensors.getLateralSpeed());
			} else if (field.equals("speedZ")) {
				x.add(sensors.getZSpeed());
			} else if (field.equals("track")) {
				x.addAll(Arrays.asList(ArrayUtils.toObject(sensors.getTrackEdgeSensors())));
			} else if (field.equals("trackPos")) {
				x.add(sensors.getTrackPosition());
			} else if (field.equals("wheelSpinVel")) {
				x.addAll(Arrays.asList(ArrayUtils.toObject(sensors.getWheelSpinVelocity())));
			} else if (field.equals("z")) {
				x.add(sensors.getZ());
			}
		}
    }
	
	public double predict(SensorModel sensors) {
		ArrayList<Double> xarray = new ArrayList<Double>();
		this.AddInputs(xarray, sensors);
		RealMatrix x = new Array2DRowRealMatrix(
				ArrayUtils.toPrimitive(xarray.toArray(new Double[xarray.size()]))
				);
		x = x.subtract(this.mean);
		
		for(int i = 0; i < x.getRowDimension(); i++) {
			x.setEntry(i, 0, x.getEntry(i, 0) / this.std.getEntry(i, 0));
		}
		
		return this.predict(x);
	}
	
	public double predict(RealMatrix x) {
		x = this.AppendOne(x);
		x = this.w_hidden_in.multiply(x);
		
		RealMatrix z = this.actv_funcs.get(0).evaluate(x); 
	    z = this.AppendOne(z);
	    
	    RealMatrix y = this.w_out_hidden.multiply(z);
	    y = this.actv_funcs.get(1).evaluate(y);
	    return y.getEntry(0, 0);
	}
	
	private double[] StringsToDoubles(String[] strarr) {
		ArrayList<Double> output = new ArrayList<>();
		for(String str : strarr) {
			output.add(Double.parseDouble(str));
		}
		return ArrayUtils.toPrimitive(output.toArray(new Double[output.size()]));
	}

	public void ReadFile(String filename) {
		this.clear();
		
		try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
			String line;
			
			// Fields
			line = br.readLine().trim();
			this.fields = line.split(WhitespaceRegex);
			br.readLine();
			
			// Hidden layer's activation function
			line = br.readLine().trim();
			this.actv_funcs.add(ActivationMap.get(line));
			br.readLine();
			
			// Output layer's activation function
			line = br.readLine().trim();
			this.actv_funcs.add(ActivationMap.get(line));
			br.readLine();
			
			// Mean
			line = br.readLine().trim();
			this.mean = new Array2DRowRealMatrix(StringsToDoubles(line.split(WhitespaceRegex)));
			br.readLine();
			
			// Standard Deviation
			line = br.readLine().trim();
			this.std = new Array2DRowRealMatrix(StringsToDoubles(line.split(WhitespaceRegex)));
			
			// Get rid of all empty lines
			while((line = br.readLine().trim()).isEmpty());
			
			ArrayList<ArrayList<Double>> w_hidden_in = new ArrayList<>();
			ArrayList<ArrayList<Double>> w_out_hidden = new ArrayList<>();
			ArrayList<ArrayList<Double>> current_w_arr = w_hidden_in;
			
			while (line != null) {
				line = line.trim();
				if (!line.isEmpty()) {
					current_w_arr.add(new ArrayList<Double>());
					String[] linevalues = line.split(WhitespaceRegex);
					for (String value : linevalues) {
						current_w_arr.get(current_w_arr.size() - 1).add(Double.parseDouble(value));
					}
				} else
					current_w_arr = w_out_hidden;
				line = br.readLine();
			}
			br.close();
			
			current_w_arr = w_hidden_in;
			for(int i = 0; i < 2; i++) {
				double[][] current_w = new double[current_w_arr.size()][];
				ArrayList<Double> current_row_arr = null;
				
				for(int j = 0; j < current_w.length; j++) {
					current_row_arr = current_w_arr.get(j);
					current_w[j] =  ArrayUtils.toPrimitive(current_row_arr.toArray(new Double[current_row_arr.size()]));
				}
				
				if(this.w_hidden_in == null) {
//					for(int j = 0; j < current_w.length; j++) {
//						for(int k = 0; k < current_w[j].length; k++)
//							System.out.print(String.valueOf(current_w[j][k]) + " ");
//						System.out.print("\n");
//					}
					this.w_hidden_in = new Array2DRowRealMatrix(current_w);
					this.w_hidden_in = this.w_hidden_in.transpose();
				} else {
//					for(int j = 0; j < current_w.length; j++) {
//						for(int k = 0; k < current_w[j].length; k++)
//							System.out.print(String.valueOf(current_w[j][k]) + " ");
//						System.out.print("\n");
//					}
					this.w_out_hidden = new Array2DRowRealMatrix(current_w);
					this.w_out_hidden = this.w_out_hidden.transpose();
				}
				
				current_w_arr = w_out_hidden;
			}
			
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public double predict_normalization(RealMatrix x, boolean normalize) {
		
		if (normalize){
			x = x.subtract(this.mean);
			
			for(int i = 0; i < x.getRowDimension(); i++) {
				x.setEntry(i, 0, x.getEntry(i, 0) / this.std.getEntry(i, 0));
			}
		}
		
		x = this.AppendOne(x);
		x = this.w_hidden_in.multiply(x);
		
		RealMatrix z = this.actv_funcs.get(0).evaluate(x); 
	    z = this.AppendOne(z);
	    
	    RealMatrix y = this.w_out_hidden.multiply(z);
	    y = this.actv_funcs.get(1).evaluate(y);
	    return y.getEntry(0, 0);
	}
	
}
