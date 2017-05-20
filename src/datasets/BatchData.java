package datasets;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import datastructs.DataSequence;
import datastructs.DataSet;
import datastructs.DataStep;
import loss.LossSumOfSquares;
import model.Model;
import model.Nonlinearity;
import model.TanhUnit;

public class BatchData extends DataSet {

	
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
	
	public BatchData(Random r, int batchsize, int nbatches) {
		
		inputDimension = 1;
		outputDimension = 1;
		
		lossTraining = new LossSumOfSquares();
		lossReporting = new LossSumOfSquares();
		
		try {
			training = generateSequences(r, batchsize, nbatches);
			validation = generateSequences(r, batchsize, nbatches);
			testing = generateSequences(r, batchsize, nbatches);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	
	
	private List<DataSequence> generateSequences(Random r, int batchsize, int nbatches) throws Exception {
		
		List<DataSequence> mydata = new ArrayList<>();
		
		double[] input = new double[batchsize];
		double[] target = new double[batchsize];
			
		for(int i = 0; i < nbatches; i++)
		{
		  for(int j = 0; j < batchsize; j++)
		  {
			input[j] = -1.0 + 2*r.nextDouble();
			target[j] = Math.sin(input[j]*Math.PI);
		  }				
		  List<DataStep> element = new ArrayList<>();
		  element.add(new DataStep(input, target, 1, batchsize));
		  mydata.add(new DataSequence(element)); 
		}
		
		return mydata;
		
	}
	
	
	
	@Override
	public void DisplayReport(Model model, Random rng) throws Exception {
		// TODO Auto-generated method stub
		
	}

	@Override
	public Nonlinearity getModelOutputUnitToUse() {
		return new TanhUnit();
	}

}
