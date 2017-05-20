

import java.util.Random;

import model.Model;
import trainer.Trainer;
import util.NeuralNetworkHelper;
import datasets.BatchData;
import datastructs.DataSet;

public class TestFeedForward {
	
	public static void main(String[] args) throws Exception {

		Random rng = new Random();
		
		int batchsize = 1;
		int nbatches = 1000; 
		
		DataSet data = new BatchData(rng, batchsize, nbatches);
					
		int hiddenDimension = 20;
		int hiddenLayers = 1;
		double learningRate = 0.004;
		double initParamsStdDev = 0.08;

		Model nn = NeuralNetworkHelper.makeFeedForward( 
				data.inputDimension, hiddenDimension, hiddenLayers, 
				data.outputDimension, data.getModelOutputUnitToUse(), 
				data.getModelOutputUnitToUse(), initParamsStdDev, rng);
		
		int reportEveryNthEpoch = 10;
		int trainingEpochs = 1000;
		
		Trainer.train(trainingEpochs, learningRate, nn, data, reportEveryNthEpoch, rng);
		
		System.out.println("done.");
	}
}


