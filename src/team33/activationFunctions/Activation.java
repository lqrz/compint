package team33.activationFunctions;

import org.apache.commons.math3.linear.RealMatrix;

public abstract class Activation {
	public abstract double evaluate(double x);
	
	public RealMatrix evaluate(RealMatrix x) {
		RealMatrix result = x.copy();
		for(int i = 0; i < x.getRowDimension(); i++)
			for(int j = 0; j < x.getColumnDimension(); j++)
				result.setEntry(i, j, this.evaluate(result.getEntry(i, j)));
		return result;
	}
}
