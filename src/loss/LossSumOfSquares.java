package loss;
import matrix.Matrix;

public class LossSumOfSquares implements Loss {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	@Override
	public void backward(Matrix actualOutput, Matrix targetOutput) throws Exception {
		for (int i = 0; i < targetOutput.w.length; i++) {
			double errDelta = actualOutput.w[i] - targetOutput.w[i];
			actualOutput.dw[i] += errDelta;
		}
		
//		System.out.println("Backward error...");
//		for (int i = 0; i < targetOutput.w.length; i++) {
//			System.out.print(actualOutput.dw[i] + ", ");
//		}
//		System.out.println("");
	}
	
	@Override
	public double measure(Matrix actualOutput, Matrix targetOutput) throws Exception {
		double sum = 0;
		for (int i = 0; i < targetOutput.w.length; i++) {
			double errDelta = actualOutput.w[i] - targetOutput.w[i];
			sum += 0.5 * errDelta * errDelta;
		}
		return sum;
	}
}
