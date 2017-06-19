package datasets;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
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

public class LODdata extends DataSet {

	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public LODdata(int number_sensors, int number_sequences, String root) throws IOException {
		
		inputDimension = number_sensors;
		outputDimension = 2;
		
		lossTraining = new LossSoftmax();
		lossReporting = new LossSoftmax();
		
		String train = root + "data/car_driverdoor.csv";
		//buildLODdata(number_sequences, 0, train);
		//training = readChunkOfData(new File(train), inputDimension);
		training = readChunkOfCarData(new File(train), inputDimension);
		
		String valid = root + "data/car_passbydriverside.csv";
		//buildLODdata(number_sequences, 563, valid);		
		validation = readChunkOfCarData(new File(valid), inputDimension);
		
		String test = root + "data/car_idle.csv";
		//buildLODdata(number_sequences, 123, test);
		testing = readChunkOfCarData(new File(valid), inputDimension);
		
	}
	
	
	public List<DataSequence> readChunkOfData(File file, int number_of_sensors) throws IOException {
		
		String delims = "[,]+";
		String[] tokens; 
		String strline; 
		List<DataSequence> result = new ArrayList<>();
		ArrayList<ArrayList<Double>> sensorCollection = new ArrayList<ArrayList<Double>>();
		
		int sensor_count = 0; 
		
		FileInputStream fin; DataInputStream din; BufferedReader br;
		fin = new FileInputStream(file);
        din = new DataInputStream(fin);
        br = new BufferedReader(new InputStreamReader(din));
		
        while((strline = br.readLine()) != null) {
        	
        	tokens = strline.split(delims);
        	
        	ArrayList<Double> sensor_read = new ArrayList<Double>();
        	for(int j = 2; j < tokens.length; j++) {
        		sensor_read.add(new Double(tokens[j]));
        	}
        	sensorCollection.add(sensor_read);
        	sensor_count++;
        	
            if(sensor_count == number_of_sensors) {
            	            
        		double[][] sensorMatrix = sensorCollection.stream().map(  u  ->  u.stream().mapToDouble(i->i).toArray()  ).toArray(double[][]::new);
        		DataSequence sequence = new DataSequence();
//        		
//        		System.out.println(sensorMatrix[0][tokens.length-3] + " " + 
//        				sensorMatrix[1][tokens.length-3] + " " + 
//        				sensorMatrix[2][tokens.length-3] + " " +
//        				sensorMatrix[3][tokens.length-3] + " " +
//        				sensorMatrix[4][tokens.length-3] + " " +
//        				sensorMatrix[5][tokens.length-3] + " " +
//        				sensorMatrix[6][tokens.length-3] + " " +
//        				sensorMatrix[7][tokens.length-3]);
        		
        		for(int j = 2; j < tokens.length; j++) {
        			
        			DataStep step = new DataStep();
        			double[] input = new double[number_of_sensors];
        			
        			for(int i = 0; i < number_of_sensors; i++) {
        				input[i] = sensorMatrix[i][j-2];
        			}
        			step.input = new Matrix(input);
        			
        			//if at end of sequence, add a target state
        			if(j == tokens.length-1) {
        				
        				double[] targetOutput = new double[2];
                		targetOutput[0] = 1.0; targetOutput[1] = 0.0;
                		                		
                		if(sensorMatrix[0][sensorMatrix[0].length-1] != 0) {
                			targetOutput[0] = 0.0; targetOutput[1] = 1.0;              			
                		}
                		//System.out.println(targetOutput[0] + " " + targetOutput[1]);
                		step.targetOutput = new Matrix(targetOutput);
        			}
        			sequence.steps.add(step);
        		}
        		
        		sensorCollection.clear();
        		sensor_count = 0;
        		result.add(sequence);	
            }
        }		
		br.close();
		return result; 
	}
	
	public static List<DataSequence> readChunkOfCarData(File file, int number_of_sensors) throws IOException {
		
		String delims = "[;]+";
		String[] tokens; 
		String strline; 
		List<DataSequence> result = new ArrayList<>();
		ArrayList<ArrayList<Double>> sensorCollection = new ArrayList<ArrayList<Double>>();
		
		int sensor_count = 0; 
		
		FileInputStream fin; DataInputStream din; BufferedReader br;
		fin = new FileInputStream(file);
        din = new DataInputStream(fin);
        br = new BufferedReader(new InputStreamReader(din));
		
        while((strline = br.readLine()) != null) {
        	
        	tokens = strline.split(delims);
        	
        	ArrayList<Double> sensor_read = new ArrayList<Double>();
        	for(int j = 2; j < tokens.length; j++) {
        		sensor_read.add((new Double(tokens[j]))/2000.0);
        	}
        	sensorCollection.add(sensor_read);
        	sensor_count++;
        	
            if(sensor_count == number_of_sensors-1) {
            	
            	//add one more to make 8
            	sensorCollection.add(sensor_read);
            	
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
        			if(j == tokens.length-1) {
        				
        				double[] targetOutput = new double[2];
                		targetOutput[0] = 1.0; targetOutput[1] = 0.0;
                		                		
                		if(sensorMatrix[0][sensorMatrix[0].length-1] != 0) {
                			targetOutput[0] = 0.0; targetOutput[1] = 1.0;              			
                		}
                		//System.out.println(targetOutput[0] + " " + targetOutput[1]);
                		step.targetOutput = new Matrix(targetOutput);
        			}
        			sequence.steps.add(step);
        		}
        		
        		sensorCollection.clear();
        		sensor_count = 0;
        		result.add(sequence);	
            }
        }		
		br.close();
		return result; 
	}	
	
	
	
	public static double[][] readNoObjectData() throws IOException {
		
		String delims = "[,]+";
		String[] tokens; 
		String strline; 
	
		File file = new File("data/nondetect.csv"); 
		
		FileInputStream fin; DataInputStream din; BufferedReader br;
		fin = new FileInputStream(file);
        din = new DataInputStream(fin);
        br = new BufferedReader(new InputStreamReader(din));
        
        double[][] data = new double[3][154];
        int count = 0; 
        
        while((strline = br.readLine()) != null) {
        	
        	if(count == 154) break;
        	
        	tokens = strline.split(delims);
        	for(int i = 0; i < tokens.length; i++) {
        		data[i][count] = (new Double(tokens[i])).doubleValue()/20000;
        	}     
        	count++;
        }
        br.close();
        
        return data;
	}
	
	public static double[][] readDetectObjectData() throws IOException {
		
		String delims = "[,]+";
		String[] tokens; 
		String strline; 
	
		File file = new File("data/detect.csv"); 
		
		FileInputStream fin; DataInputStream din; BufferedReader br;
		fin = new FileInputStream(file);
        din = new DataInputStream(fin);
        br = new BufferedReader(new InputStreamReader(din));
        
        double[][] data = new double[4][154];
        int count = 0; 
        
        while((strline = br.readLine()) != null) {
        	
        	if(count == 154) break;
        	
        	tokens = strline.split(delims);
        	for(int i = 0; i < tokens.length; i++) {
        		data[i][count] = (new Double(tokens[i])).doubleValue()/20000;
        	}     
            count++;
        }
        br.close();
        
        return data;
	}
	
	public static void buildLODdata(int number_samples, long seed, String file) throws IOException {
		
		Random grd = new Random(seed);
		int which = 0;
		int state = 0;
		double prob_state = 0;
		double val = 0; 
		
		java.util.Date date = new java.util.Date();
		PrintWriter output = new PrintWriter(file);
		
		double[][] noLOD = readNoObjectData();
		double[][] LOD = readDetectObjectData();
		
		for(int i = 0; i < number_samples; i++) {
			
			prob_state = grd.nextDouble();
			
			state = 2; 
			if(prob_state < .6) state = 0;
			else if(prob_state > .6 && prob_state < .9) state = 1;
							
			for(int j = 0; j < 8; j++) {

				//System.out.println(j + ", " + state);
				output.print(j + ", " + date.getTime() + ", ");
				if(state == 0) { //create no-detect data
					
					which = grd.nextInt(3);
					for(int k = 0; k < 154; k++) {
						val = Math.abs(noLOD[which][k] + grd.nextGaussian()*.1);
						output.print(val + ", ");
					}
					output.println("" + 0);	
				}
				else if(state == 1) { //create detect data
					
				    which = grd.nextInt(4);
					for(int k = 0; k < 154; k++) {
						val = Math.abs(LOD[which][k] + grd.nextGaussian()*.1);
						output.print(val + ", ");
					}
					output.println("" + 1);	
				}
				else if(state == 2) { //create mix data
					
					which = grd.nextInt(7);
					for(int k = 0; k < 154; k++) {
						if(which < 3) val = Math.abs(noLOD[which][k] + grd.nextGaussian()*.1);
						else val = Math.abs(LOD[which - 3][k] + grd.nextGaussian()*.1);
						output.print(val + ", ");
					}
					output.println("" + 1);	
				}	
			}
		}
		output.close();
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
