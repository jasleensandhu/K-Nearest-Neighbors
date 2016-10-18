package KNN;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;

public class KNNDriver {


	public static void main(String[] args) throws FileNotFoundException {
		//Get output in ConsoleOutput.txt file
		PrintStream out = new PrintStream(new FileOutputStream("ConsoleOutput.txt")); 
		System.setOut(out);

		KNearest kFold = new KNearest("C:\\UTA\\Fall2016\\Machine Learning\\Assn2\\Yeast.csv",10);
		
		kFold.ReadDataset();
		
		//int k = 10;
		
		kFold.KFoldCrossValidation();
		kFold.SetTestingDataset();

		kFold.nearestDistance(0,7); //calculate based on eucledian
		kFold.nearestDistance(1,7); //calculate based on Polynomial Kernel
		kFold.nearestDistance(2,7); //calculate based on Polynomial Kernel
		
		
	
	}
	
}
