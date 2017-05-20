package datasets;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import datastructs.DataSequence;
import datastructs.DataSet;
import datastructs.DataStep;
import matrix.Matrix;
import model.Model;
import model.Nonlinearity;
import model.SigmoidUnit;

public class LODdata extends DataSet {

	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private List<DataSequence> readChunkOfData(File file, int number_of_sensors, int total_sequences) throws IOException {
		
		String delims = "[,]+";
		String[] tokens; 
		String strline; 
		List<DataSequence> result = new ArrayList<>();
		ArrayList<ArrayList<Double>> sensorCollection = new ArrayList<ArrayList<Double>>();
		
		int sensor_count = 0; Double D;
		
		FileInputStream fin; DataInputStream din; BufferedReader br;
		fin = new FileInputStream(file);
        din = new DataInputStream(fin);
        br = new BufferedReader(new InputStreamReader(din));
		
        while((strline = br.readLine()) != null) {
        	
        	tokens = strline.split(delims);
        	
        	if(sensor_count < number_of_sensors) {
        		
        		ArrayList<Double> sensor_read = new ArrayList<Double>();
        		for(int j = 2; j < tokens.length; j++) {
        			sensor_read.add(new Double(tokens[j]));
        		}
        		sensorCollection.add(sensor_read);
        		sensor_count++;
        	}
        	else {
        		
        		double[][] sensorMatrix = sensorCollection.stream().map(  u  ->  u.stream().mapToDouble(i->i).toArray()  ).toArray(double[][]::new);
        		DataSequence sequence = new DataSequence();
        		
        		for(int j = 2; j < tokens.length; j++) {
        			
        			DataStep step = new DataStep();
        			double[] input = new double[number_of_sensors];
        			
        			for(int i = 0; i < number_of_sensors; i++) {
        				input[i] = sensorMatrix[i][j-2];
        			}
        			step.input = new Matrix(input);
        			
        			//if at end of sequence, add a target state
        			if(j == tokens.length-2) {
        				
        				double[] targetOutput = new double[2];
                		targetOutput[0] = 1.0; targetOutput[1] = 0.0;
                		
                		if(sensorMatrix[0][sensorMatrix[0].length-1] != 0) {
                			targetOutput[0] = 0.0; targetOutput[1] = 1.0;
                		}
                		step.targetOutput = new Matrix(targetOutput);
        			}
        			sequence.steps.add(step);
        		}
        		        		
        		sensor_count = 0;
        		result.add(sequence);
        	}	
        }
        
		
		br.close();
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
