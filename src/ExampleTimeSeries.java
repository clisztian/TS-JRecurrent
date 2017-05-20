
import java.util.Random;

import model.Model;
import trainer.Trainer;
import util.NeuralNetworkHelper;
import datasets.MultivariateDynamicTS;
import datasets.MultivariateTS;
import datastructs.DataSet;

public class ExampleTimeSeries {
	
	public static void main(String[] args) throws Exception {

		Random rng = new Random();
		
		int number_sequences = 1000;
		int max_sequence_length_train = 200;
		int max_sequence_length_test = 200;
		int nclasses = 3;
		
		int batchsize = 10;
		
		//DataSet data = new MultivariateTS(rng, number_sequences, max_sequence_length_train, max_sequence_length_test, nclasses);
		
		DataSet data = new MultivariateDynamicTS(rng, number_sequences, max_sequence_length_train, max_sequence_length_test);
		
		int hiddenDimension = 20;
		int hiddenLayers = 2;
		double learningRate = 0.004;
		double initParamsStdDev = 0.08;

		Model nn = NeuralNetworkHelper.makeGru( 
				data.inputDimension,
				hiddenDimension, hiddenLayers, 
				data.outputDimension, data.getModelOutputUnitToUse(), 
				initParamsStdDev, rng);
		
//		Model nn = NeuralNetworkHelper.makeBatchMGU( 
//				data.inputDimension,
//				hiddenDimension, batchsize, hiddenLayers, 
//				data.outputDimension, data.getModelOutputUnitToUse(), 
//				initParamsStdDev, rng);
		
		int reportEveryNthEpoch = 10;
		int trainingEpochs = 1000;
		
		Trainer.train(trainingEpochs, learningRate, nn, data, reportEveryNthEpoch, rng);
		
		System.out.println("done.");
	}
}

