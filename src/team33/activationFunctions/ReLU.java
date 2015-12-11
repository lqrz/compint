package team33.activationFunctions;

public class ReLU extends Activation {
	@Override
	public double evaluate(double x) {
		return Math.max(0.0, x);
	}

}
