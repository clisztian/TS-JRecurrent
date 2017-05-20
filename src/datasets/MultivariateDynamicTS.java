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

public class MultivariateDynamicTS extends DataSet {

	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	double mean1 = 0; 
	double mean2 = 2;
	double mean3 = -2;
	double std1 = 1.0;
	double std2 = 3.0;
	double std3 = 2.0;	
	int nClasses = 3;
	
	public MultivariateDynamicTS(Random r, int total_sequences, int max_sequence_length_train, int max_sequence_length_test) {
		
		inputDimension = 3;
		outputDimension = 2;
		
		lossTraining = new LossSoftmax();
		lossReporting = new LossSoftmax();
		training = generateSequences(r, total_sequences, max_sequence_length_train);
		validation = generateSequences(r, total_sequences, max_sequence_length_train);
		testing = generateSequences(r, total_sequences, max_sequence_length_test);
		unit = generateUnitTestSequence(r, 3, max_sequence_length_test);
	}

	public MultivariateDynamicTS(Random r, int total_sequences, int batchsize, int max_sequence_length_train, int max_sequence_length_test) throws Exception {
		
		inputDimension = 3;
		outputDimension = 2;
		
		lossTraining = new LossSoftmax();
		lossReporting = new LossSoftmax();
		training = generateSequencesBatch(r, batchsize, total_sequences, max_sequence_length_train);
		validation = generateSequencesBatch(r, batchsize, total_sequences, max_sequence_length_train);
		testing = generateSequencesBatch(r, batchsize, total_sequences, max_sequence_length_test);
		
	}	
	
	
	private List<DataSequence> generateSequences(Random r, int total_sequences, int max_sequence_length) {
		
		double ar1 = .2; 
		double ar2 = .7; 
		double ar3 = .9;
		boolean normal = true;
		double action = 1.0;
		
		List<DataSequence> result = new ArrayList<>();
		for (int s = 0; s < total_sequences; s++) {
			
			normal = true;
			DataSequence sequence = new DataSequence();
			//int tempSequenceLength = r.nextInt(max_sequence_length) + 1;
			int tempSequenceLength = max_sequence_length;
			int classMe = r.nextInt(30) + 30;
			
			double y = .01;
			double y1 = .03;
			double y2 = .02;
			
			for (int t = 0; t < tempSequenceLength; t++) {
				
				DataStep step = new DataStep();
				action = 1.0;
				
				if(t%classMe == 0) normal = !normal;
				if(!normal) action = 5.0;
				
				double[] input = new double[inputDimension];				
				input[0] = -ar1*y + r.nextGaussian()*std1*action + mean1;
				input[1] = -ar2*y1 + r.nextGaussian()*std2*action + mean2; 
				input[2] = -ar3*y2 + r.nextGaussian()*std3*action + mean3; 
				
				y = input[0];
				y1 = input[1];
				y2 = input[2];

				step.input = new Matrix(input);
				
				double[] targetOutput = new double[2];
				
				if(!normal) {
					targetOutput[1] = 1.0;
				}
				else {
					targetOutput[0] = 1.0;
				}
				step.targetOutput = new Matrix(targetOutput);				
				sequence.steps.add(step);
			}
			result.add(sequence);
		}
		return result;
	}
	
	private List<DataSequence> generateSequencesBatch(Random r, int batchsize, int total_sequences, int max_sequence_length) throws Exception {
		
		double ar1 = .2; 
		double ar2 = .7; 
		double ar3 = .9;
		boolean normal = true;
		double action = 1.0;
		
		List<DataSequence> result = new ArrayList<>();
		for (int s = 0; s < total_sequences; s++) {
			
			normal = true;
			DataSequence sequence = new DataSequence();
			//int tempSequenceLength = r.nextInt(max_sequence_length) + 1;
			int tempSequenceLength = max_sequence_length;
			int classMe = r.nextInt(30) + 30;
			
			double y = .01;
			double y1 = .03;
			double y2 = .02;
			
			for (int t = 0; t < tempSequenceLength; t++) {
				
				DataStep step = new DataStep();
				action = 1.0;
				
				if(t%classMe == 0) normal = !normal;
				if(!normal) action = 5.0;
				
				double[] input = new double[inputDimension*batchsize];
				double[] targetOutput = new double[outputDimension*batchsize];
				
				for(int j = 0; j < batchsize; j++) {
					
					input[0*batchsize + j] = -ar1*input[0*batchsize + j] + r.nextGaussian()*std1*action + mean1;
					input[1*batchsize + j] = -ar2*input[1*batchsize + j] + r.nextGaussian()*std2*action + mean2; 
					input[2*batchsize + j] = -ar3*input[2*batchsize + j] + r.nextGaussian()*std3*action + mean3; 
											    
				
				    if(!normal) {
					  targetOutput[batchsize + j] = 1.0;
				    }
				    else {
					  targetOutput[j] = 1.0;
				    }
				    
				}
				
				step.input = new Matrix(input, inputDimension, batchsize);
				step.targetOutput = new Matrix(targetOutput, outputDimension, batchsize);				
			    sequence.steps.add(step);
			}
			result.add(sequence);
		}
		return result;
	}
	
	
	
	
    private List<DataSequence> generateUnitTestSequence(Random r, int total_sequences, int max_sequence_length) {
		
		double ar1 = .2; 
		double ar2 = .7; 
		double ar3 = .9;
		
		
		List<DataSequence> result = new ArrayList<>();
		for (int s = 0; s < total_sequences; s++) {
			
		
			DataSequence sequence = new DataSequence();
			int tempSequenceLength = r.nextInt(max_sequence_length) + 1;
			double y = .01;
			double y1 = .03;
			double y2 = .02;
			
			for (int t = 0; t < tempSequenceLength; t++) {
				
				DataStep step = new DataStep();

				double[] input = new double[inputDimension];				
				input[0] = -ar1*y + r.nextGaussian()*std1 + mean1;
				input[1] = -ar2*y1 + r.nextGaussian()*std2 + mean2; 
				input[2] = -ar3*y2 + r.nextGaussian()*std3 + mean3; 
				
				y = input[0];
				y1 = input[1];
				y2 = input[2];

				step.input = new Matrix(input);
				
				double[] targetOutput = new double[2];
				targetOutput[0] = 1.0;

				step.targetOutput = new Matrix(targetOutput);				
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
