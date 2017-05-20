
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import autodiff.Graph;
import loss.Loss;
import loss.LossMultiDimensionalBinary;
import loss.LossSumOfSquares;
import matrix.Matrix;
import model.NeuralNetwork;
import model.Nonlinearity;
import model.SigmoidUnit;
import trainer.Trainer;
import util.NeuralNetworkHelper;


public class ExampleTestLSTM {

	public static void main(String[] args) throws Exception {

		Random rng = new Random();
		boolean applyTraining = true;
		Graph g = new Graph(applyTraining);
		

		int inputDimension = 6;
		int outputDimension = 6;
		
		double[] samp = new double[inputDimension];
		double[] target = new double[outputDimension];
		samp[1] = 1.0; target[5] = 1.0;
		Matrix input = new Matrix(samp);
		Matrix targetOutput = new Matrix(target);
		
		double[] samp2 = new double[inputDimension];
		double[] target2 = new double[outputDimension];
		samp2[0] = 1.0; target2[2] = 1.0;
		Matrix input1 = new Matrix(samp2);
		Matrix targetOutput1 = new Matrix(target2);
		
        List<Matrix> inputs = new ArrayList<>();
        inputs.add(input); inputs.add(input1);
		
        List<Matrix> targets = new ArrayList<>();
        targets.add(targetOutput); targets.add(targetOutput1);
		

		int hiddenDimension = 7;
		int hiddenLayers = 1;
		double learningRate = 0.001;
		double initParamsStdDev = 0.08;
        		
		
		
		Nonlinearity decoder = new SigmoidUnit();
		Loss lossReporting = new LossMultiDimensionalBinary();
		Loss lossTraining = new LossSumOfSquares();
		
		
		
		NeuralNetwork nn = NeuralNetworkHelper.makeLstm( 
				inputDimension, 
				hiddenDimension, hiddenLayers, 
				outputDimension, decoder, 
				initParamsStdDev, rng);
		

		nn.resetState();
		
		for(int i = 0; i < 2; i++)
		{
		 
			Matrix output = nn.forward(inputs.get(i), g);
				
		 //nn.forward_ff(input, g);
		
			output.printMatrix();
		
		    double loss = lossReporting.measure(output, targets.get(i));
		    //double loss = lossReporting.measure(nn.getOutput(), targetOutput);
	        System.out.println("Loss = " + loss);
			
		    lossTraining.backward(output, targets.get(i));
		}
		
		
		
		
		//nn.getOutput().printMatrix();
		
		//System.out.println("Loss = " + loss);
		

		//nn.getOutput().printMatrixDW();
		
        g.backward(); 
        
        //LstmLayer temp = (LstmLayer) nn.layers.get(0);
        //temp.printOutputWih();
        
        
        Trainer.updateModelParams(nn, learningRate);


        List<Matrix> params = nn.getParameters();

        for(int i = 0; i < params.size(); i++)
        {
        	System.out.println("Parameters " + i);
        	params.get(i).printMatrix();
        }
        
        
        
        
		System.out.println("done.");
		

		
		
		
	}
}
	
	

