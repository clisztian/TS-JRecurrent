package trainer;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import util.FileIO;
import autodiff.Graph;
import datastructs.DataSequence;
import datastructs.DataSet;
import datastructs.DataStep;
import loss.Loss;
import matrix.Matrix;
import model.Model;

public class Trainer {
	
	public static double decayRate = 0.999;
	public static double smoothEpsilon = 1e-8;
	public static double gradientClipValue = 5;
	public static double regularization = 0.000001; // L2 regularization strength
	
	public static double train(int trainingEpochs, double learningRate, Model model, DataSet data, int reportEveryNthEpoch, Random rng) throws Exception {
		return train(trainingEpochs, learningRate, model, data, reportEveryNthEpoch, false, false, null, rng);
	}
	
	public static double train(int trainingEpochs, double learningRate, Model model, DataSet data, int reportEveryNthEpoch, boolean initFromSaved, boolean overwriteSaved, String savePath, Random rng) throws Exception {
		System.out.println("--------------------------------------------------------------");
		
		if (initFromSaved) {
			System.out.println("initializing model from saved state...");
			try {
				model = (Model)FileIO.deserialize(savePath);
				data.DisplayReport(model, rng);
			}
			catch (Exception e) {
				System.out.println("Oops. Unable to load from a saved state.");
				System.out.println("WARNING: " + e.getMessage());
				System.out.println("Continuing from freshly initialized model instead.");
			}
		}
		double result = 1.0;
		for (int epoch = 0; epoch < trainingEpochs; epoch++) {
			
			String show = "epoch["+(epoch+1)+"/"+trainingEpochs+"]";
			
//			if (data.validation != null && epoch == 0 ) {
//				double reportedLossValidation = passScore(learningRate, model, data.validation, false, data.lossTraining, data.lossReporting);
//				
//			}
			
			
			double reportedLossTrain = pass(learningRate, model, data.training, true, data.lossTraining, data.lossReporting);
			result = reportedLossTrain;
			if (Double.isNaN(reportedLossTrain) || Double.isInfinite(reportedLossTrain)) {
				throw new Exception("WARNING: invalid value for training loss. Try lowering learning rate.");
			}
			double reportedLossValidation = 0;
			double reportedLossTesting = 0;
			
			
			
			if (data.validation != null && epoch%reportEveryNthEpoch == reportEveryNthEpoch - 1 ) {
				reportedLossValidation = passScore(learningRate, model, data.validation, false, data.lossTraining, data.lossReporting);
				result = reportedLossValidation;
			}
			if (data.testing != null) {
				reportedLossTesting = pass(learningRate, model, data.testing, false, data.lossTraining, data.lossReporting);
				result = reportedLossTesting;
			}
			if (data.unit != null && epoch%reportEveryNthEpoch == reportEveryNthEpoch - 1 ) {
				passEval(learningRate, model, data.validation, false, data.lossTraining, data.lossReporting);
			}
			
			show += "\ttrain loss = "+String.format("%.5f", reportedLossTrain);
			if (data.validation != null) {
				show += "\tvalid loss = "+String.format("%.5f", reportedLossValidation);
			}
			if (data.testing != null) {
				show += "\ttest loss  = "+String.format("%.5f", reportedLossTesting);
			}
			System.out.println(show);
			
			if (epoch % reportEveryNthEpoch == reportEveryNthEpoch - 1) {
				data.DisplayReport(model, rng);
			}
			
			if (overwriteSaved) {
				FileIO.serialize(savePath, model);
			}
			
			if (reportedLossTrain == 0 && reportedLossValidation == 0) {
				System.out.println("--------------------------------------------------------------");
				System.out.println("\nDONE.");
				break;
			}
		}
		return result;
	}
	
	public static double pass(double learningRate, Model model, List<DataSequence> sequences, boolean applyTraining, Loss lossTraining, Loss lossReporting) throws Exception {
		
		double numerLoss = 0;
		double denomLoss = 0;
		
		for (DataSequence seq : sequences) {
			model.resetState();
			Graph g = new Graph(applyTraining);
			
			for (DataStep step : seq.steps) {
				Matrix output = model.forward(step.input, g);				
				if (step.targetOutput != null) {
					double loss = lossReporting.measure(output, step.targetOutput);
					if (Double.isNaN(loss) || Double.isInfinite(loss)) {
						return loss;
					}
					numerLoss += loss;
					denomLoss++;			
					if (applyTraining) {
						lossTraining.backward(output, step.targetOutput);
					}
				}
			}
			if (applyTraining) {
				g.backward(); //backprop dw values
				updateModelParams(model, learningRate); //update params
			}	
		}
		return numerLoss/denomLoss;
	}


	
	
	public static double passScore(double learningRate, Model model, List<DataSequence> sequences, boolean applyTraining, Loss lossTraining, Loss lossReporting) throws Exception {
		
		double numerLoss = 0;
		double denomLoss = 0;
		int correctScore = 0;
		int total = 0;
		for (DataSequence seq : sequences) {
			
			System.out.println(seq.toString());
			model.resetState();
			Graph g = new Graph(applyTraining);
			for (DataStep step : seq.steps) {
				Matrix output = model.forward(step.input, g);				
				if (step.targetOutput != null) {
					
					double loss = lossReporting.measure(output, step.targetOutput);
//					System.out.println(step.input.toString());
//					System.out.println("==== Output " + output.toString());
					
					total++;
					if(maxindx(output.w) == maxindx(step.targetOutput.w)) {
						correctScore++;
					}
					
					
					if (Double.isNaN(loss) || Double.isInfinite(loss)) {
						return loss;
					}
					numerLoss += loss;
					denomLoss++;			
					if (applyTraining) {
						lossTraining.backward(output, step.targetOutput);
					}
				}
			}
			if (applyTraining) {
				g.backward(); //backprop dw values
				updateModelParams(model, learningRate); //update params
			}	
		}
		
		System.out.println("Correctly classified " + correctScore + " out of " + total + " in validation set");
		
		return numerLoss/denomLoss;
	}
	
	
	
    public static void passEval(double learningRate, Model model, List<DataSequence> sequences, boolean applyTraining, Loss lossTraining, Loss lossReporting) throws Exception {
		
    	int correctScore = 0;
		int total = 0;
    	
		for (DataSequence seq : sequences) {
			model.resetState();
			Graph g = new Graph(applyTraining);
			for (DataStep step : seq.steps) {
				
				Matrix output = model.forward(step.input, g);				
				
				total++;
				if(maxindx(output.w) == maxindx(step.targetOutput.w)) {
					correctScore++;
				}
				

				System.out.print(step.input.toString() + " ");
				output.printMatrix();
				System.out.println("");
				
			}
		}
		
		System.out.println("Correctly classified " + correctScore + " out of " + total + " in validation set");

	}
	
	
	
	
	
	
	
	
	
	private static int maxindx(double[] w) {
		
		int indx = 0; double max = w[0];
		for(int i = 1; i < w.length; i++) {
			if(max < w[i]) {
				indx = i; max = w[i];
			}
		}
		
		return indx;
	}

	public static void updateModelParams(Model model, double stepSize) throws Exception {
		for (Matrix m : model.getParameters()) {
			for (int i = 0; i < m.w.length; i++) {
				
				// rmsprop adaptive learning rate
				double mdwi = m.dw[i];
				m.stepCache[i] = m.stepCache[i] * decayRate + (1 - decayRate) * mdwi * mdwi;
				
				// gradient clip
				if (mdwi > gradientClipValue) {
					mdwi = gradientClipValue;
				}
				if (mdwi < -gradientClipValue) {
					mdwi = -gradientClipValue;
				}
				
				// update (and regularize)
				m.w[i] += - stepSize * mdwi / Math.sqrt(m.stepCache[i] + smoothEpsilon) - regularization * m.w[i];
				m.dw[i] = 0;
			}
		}
	}
}
