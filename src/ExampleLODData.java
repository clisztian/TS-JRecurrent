import java.io.File;
import java.util.List;
import java.util.Random;

import datasets.LODdata;
import datastructs.DataSequence;
import datastructs.DataSet;
import model.Model;
import trainer.Trainer;
import util.NeuralNetworkHelper;

public class ExampleLODData {
	
	public static void main(String[] args) throws Exception {
		

		Random rng = new Random();
		int number_sequences = 500;
		int number_sensors = 8;
		
		DataSet data = new LODdata(number_sensors, number_sequences, "");
		int hiddenDimension = 20;
		int hiddenLayers = 2;
		double learningRate = 0.004;
		double initParamsStdDev = 0.08;

		Model nn = NeuralNetworkHelper.makeGru( 
				data.inputDimension,
				hiddenDimension, hiddenLayers, 
				data.outputDimension, data.getModelOutputUnitToUse(), 
				initParamsStdDev, rng);
				
		int reportEveryNthEpoch = 10;
		int trainingEpochs = 1000;
		
		String modelLoc = "/home/lisztian/TS-JRecurrent/saved_models/LODmodel.ser";
		
		Trainer.train(trainingEpochs, learningRate, nn, data, reportEveryNthEpoch, rng, modelLoc);
		
		System.out.println("done.");
	}
	
}
