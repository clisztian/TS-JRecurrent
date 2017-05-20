package model;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import autodiff.Graph;
import matrix.Matrix;

public class MinGLayer implements Model {

	private static final long serialVersionUID = 1L;
	int inputDimension;
	int outputDimension;
	int batchsize; 
	
	Matrix Wi, Wh, b;
	Matrix IHmix, HHmix, Bmix;
	Matrix hidden;
	
	Nonlinearity fMix = new SigmoidUnit();
	Nonlinearity fNew = new TanhUnit();
	
	
	public MinGLayer(int inputDimension, int outputDimension, int batchsize, double initParamsStdDev, Random rng) {
		this.inputDimension = inputDimension;
		this.outputDimension = outputDimension;
		this.batchsize = batchsize;
		
		IHmix = Matrix.rand(outputDimension, inputDimension, initParamsStdDev, rng);
		HHmix = Matrix.rand(outputDimension, outputDimension, initParamsStdDev, rng);
		Bmix = new Matrix(outputDimension);
		Wi = Matrix.rand(outputDimension, inputDimension, initParamsStdDev, rng);
		Wh = Matrix.rand(outputDimension, outputDimension, initParamsStdDev, rng);
		b = new Matrix(outputDimension);
		
		//hidden = Matrix.rand(outputDimension, 1, initParamsStdDev, rng);
	}
	
	
	
	@Override
	public Matrix forward(Matrix input, Graph g) throws Exception {
		
		Matrix sum0 = g.mul(Wi, input);
		Matrix sum1 = g.mul(Wh, hidden);
		Matrix actMix = g.nonlin(fMix, g.addbatch(g.add(sum0, sum1), b));

		//actMix.printMatrix();
		
		Matrix context = g.elmul(actMix, hidden);				
		Matrix sum2 = g.mul(IHmix, input);
		Matrix sum3 = g.mul(HHmix, context);		
		Matrix temp = g.nonlin(fNew, g.addbatch(g.add(sum2, sum3), Bmix));		
		
		
		Matrix outputh1 = g.elmul(g.oneMinus(actMix), hidden);
		Matrix outputh2 = g.elmul(actMix, temp);
		Matrix output = g.add(outputh1, outputh2);
		
		hidden = output;
		
		return output;
	}
	
	
	@Override
	public void resetState() {
		
		hidden = new Matrix(outputDimension, batchsize);
	}
	
	@Override
	public List<Matrix> getParameters() {
		
		List<Matrix> result = new ArrayList<>();
		result.add(IHmix);
		result.add(HHmix);
		result.add(Bmix);
		result.add(Wi);
		result.add(Wh);
		result.add(b);
		
		return result;
	}
	
}