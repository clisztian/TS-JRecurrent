
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;

import autodiff.Graph;
import datasets.LODdata;
import datastructs.DataSequence;
import datastructs.DataStep;
import matrix.Matrix;
import model.Model;
import util.FileIO;

import com.pi4j.io.gpio.GpioController;
import com.pi4j.io.gpio.GpioFactory;
import com.pi4j.io.gpio.GpioPinDigitalOutput;
import com.pi4j.io.gpio.PinState;
import com.pi4j.io.gpio.RaspiPin;


public class TestLiveLOD {
	
	public static void main(String[] args) throws Exception {
		
		final GpioController gpio = GpioFactory.getInstance();
		final GpioPinDigitalOutput pin = gpio.provisionDigitalOutputPin(RaspiPin.GPIO_22, "PinLED", PinState.HIGH);
		
		
		File file = new File("/tmp/lod_data.csv");
		//String savePath = "/home/pi/TS-JRecurrent/saved_models/LODmodel.ser";
		String savePath = "/home/lisztian/TS-JRecurrent/saved_models/LODmodel.ser";
		
		Model model = (Model)FileIO.deserialize(savePath);


        
        // creating the pin with parameter PinState.HIGH
        // will instantly power up the pin
        
        System.out.println("light is: ON");
        
        // wait 2 seconds
        //Thread.sleep(2000);
        
        // turn off GPIO 1
        //pin.low();
        System.out.println("light is: OFF");
        
        TimerTask task = new TimerTask() {
            @Override
            public void run() {
            	List<DataSequence> sequences = null;
				try {
					sequences = LODdata.readChunkOfCarData(file, 8);
				} catch (IOException e1) {
					e1.printStackTrace();
				}
        		
        		for (DataSequence seq : sequences) {
        			model.resetState();
        			Graph g = new Graph(false);
        			
        			Matrix output = null;
        			for (DataStep step : seq.steps) {
        				try {
							output = model.forward(step.input, g);
						} catch (Exception e) {
							e.printStackTrace();
						}
        			}
        			output.printMatrix();
        			
//        			if(output.w[1] > .90) {
//        				pin.high();
//        			}
//        			else {
//        				pin.low();
//        			}
        		}
            }
        };
		
		

        Timer timer = new Timer();
        long delay = 0;
        long intevalPeriod = 2000; 
        
        // schedules the task to be run in an interval 
        timer.scheduleAtFixedRate(task, delay, intevalPeriod);

		//gpio.shutdown();
	}
	
	
	
	
}

