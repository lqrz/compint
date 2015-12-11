package team33.activationFunctions;

public class Sigmoid extends Activation {
	@Override
	public double evaluate(double x) {
		return 1.0 / (1 + Math.exp(-x));
	}
}
