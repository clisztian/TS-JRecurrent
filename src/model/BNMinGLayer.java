package model;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import autodiff.Graph;
import matrix.Matrix;

public class BNMinGLayer implements Model {

	private static final long serialVersionUID = 1L;
	int inputDimension;
	int outputDimension;
	int batchsize; 
	
	Matrix Wi, Wh, b;
	Matrix alpha, beta;
	Matrix alphah, betah;
	Matrix IHmix, HHmix, Bmix;
	Matrix hidden;
	
	Nonlinearity fMix = new SigmoidUnit();
	Nonlinearity fNew = new TanhUnit();
	
	
	public BNMinGLayer(int inputDimension, int outputDimension, int batchsize, double initParamsStdDev, Random rng) {
		
		this.inputDimension = inputDimension;
		this.outputDimension = outputDimension;
		this.batchsize = batchsize;
		
		alpha = new Matrix(inputDimension, batchsize);
		beta = Matrix.ones(inputDimension, batchsize);

		alphah = new Matrix(outputDimension, batchsize);
		betah = Matrix.ones(outputDimension, batchsize);
		
		
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
		
		input.normalizeBatch();
		hidden.normalizeBatch();
		
		Matrix norminput = g.add(alpha, g.elmul(beta, input));
		Matrix normhidden = g.add(alphah, g.elmul(betah, hidden));
		
		Matrix sum0 = g.mul(Wi, norminput);
		Matrix sum1 = g.mul(Wh, normhidden);
		
		Matrix actMix = g.nonlin(fMix, g.addbatch(g.add(sum0, sum1), b));
		
		Matrix context = g.elmul(actMix, normhidden);				
		Matrix sum2 = g.mul(IHmix, norminput);
		Matrix sum3 = g.mul(HHmix, context);		
		Matrix temp = g.nonlin(fNew, g.addbatch(g.add(sum2, sum3), Bmix));		
		
		
		Matrix outputh1 = g.elmul(g.oneMinus(actMix), normhidden);
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
		
		result.add(alpha);
		result.add(beta);
		result.add(alphah);
		result.add(betah);		
		result.add(IHmix);
		result.add(HHmix);
		result.add(Bmix);
		result.add(Wi);
		result.add(Wh);
		result.add(b);
		
		return result;
	}
	
}
