package team33.activationFunctions;

public class Tanh extends Activation {
	@Override
	public double evaluate(double x) {
		return (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x));
	}
}
