package datasets;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import datastructs.DataSequence;
import datastructs.DataSet;
import datastructs.DataStep;
import loss.LossSoftmax;
import matrix.Matrix;
import model.Model;
import model.Nonlinearity;
import model.SigmoidUnit;

public class MultivariateTS extends DataSet {

	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	double mean1 = 0; 
	double mean2 = 2;
	double mean3 = -2;
	double std1 = 1.0;
	double std2 = 3.0;
	double std3 = 1.0;
	
	int nClasses = 3;
	
	public MultivariateTS(Random r, int total_sequences, int max_sequence_length_train, int max_sequence_length_test, int nclasses) {
		inputDimension = 1;
		outputDimension = nclasses;
		nClasses = nclasses;
		
		lossTraining = new LossSoftmax();
		lossReporting = new LossSoftmax();
		training = generateSequences(r, total_sequences, max_sequence_length_train);
		validation = generateSequences(r, total_sequences, max_sequence_length_train);
		testing = generateSequences(r, total_sequences, max_sequence_length_test);
	}
	
	private void setDistribution(double m1, double m2, double s1, double s2) {
		
		mean1 = m1; 
		mean2 = m2;
		std1 = s1;
		std2 = s2;		
		mean3 = -mean2;
		std3 = s1;
	}
	
	private List<DataSequence> generateSequences(Random r, int total_sequences, int max_sequence_length) {
		
		double ar1 = .2; 
		double ar2 = .7; 
		double ar3 = .9;
		
		List<DataSequence> result = new ArrayList<>();
		for (int s = 0; s < total_sequences; s++) {
			
			DataSequence sequence = new DataSequence();
			int classMe = 0;
			int tempSequenceLength = r.nextInt(max_sequence_length) + 1;
			
			
			double y = .01;
			
			classMe = r.nextInt(nClasses);
			
			for (int t = 0; t < tempSequenceLength; t++) {
				
				DataStep step = new DataStep();
				double[] input = {0.0};

				if (classMe == 0) {
					input[0] = -ar1*y + r.nextGaussian()*std1 + mean1; 				
				}
				else if(classMe == 1){
					input[0] = -ar2*y + r.nextGaussian()*std2 + mean2; 
				}
				else {
					input[0] = -ar3*y + r.nextGaussian()*std3 + mean3; 
				}
				y = input[0];
				step.input = new Matrix(input);
				
				double[] targetOutput = null;
				if (t == tempSequenceLength - 1) {
					targetOutput = new double[nClasses];
					targetOutput[classMe] = 1.0;
					step.targetOutput = new Matrix(targetOutput);
				}
				sequence.steps.add(step);
			}
			result.add(sequence);
		}
		return result;
	}
	
	
	
	
	
	@Override
	public void DisplayReport(Model model, Random rng) throws Exception {
		// TODO Auto-generated method stub
		
	}

	@Override
	public Nonlinearity getModelOutputUnitToUse() {
		return new SigmoidUnit();
	}

}
