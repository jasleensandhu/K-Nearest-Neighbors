package KNN;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import org.xml.sax.HandlerBase;

public class KNearest {
	String line = "";
	String filename ;
	String csvSplitBy = ",";
	int partitionSize;

	static ArrayList<Attributes> attributeValueList = null; //attribute values not needed	
	ArrayList<Attributes[]> kFoldPartition; //partition need to check against trainingset
	Attributes[] testingDataset; //training data set



	public KNearest(String filename,int psize) { 
		partitionSize = psize;
		this.filename = filename;
	}

	//reading from file
	public void ReadDataset() {

		try {
			BufferedReader br = new BufferedReader(new FileReader(this.filename));

			// fetching the column name
			line = br.readLine();

			// column names in array
			String[] ecoliColumnName = line.split(csvSplitBy);
			int size = ecoliColumnName.length;

			for (String colName : ecoliColumnName) {
				System.out.print(colName + " ");
			}
			System.out.print("\n");

			// Fetching attributes into ArrayList
			attributeValueList = new ArrayList<Attributes>();
			while ((line = br.readLine()) != null) {
				Attributes ea = new Attributes(size, ecoliColumnName);
				ea.parseString(line);
				attributeValueList.add(ea);
				System.out.println(ea);
				System.out.println("--------------------------");
			}
			br.close();
		} catch (FileNotFoundException ex) {
			System.out.println("Unable to open csv file");
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			Attributes.printMinMax(); //max value of each attribute
			System.out.println("Normalized Dataset");
			for (Attributes ea : attributeValueList) {
				ea.Normalize();
				System.out.println("***********************");
				System.out.println(ea);
			}
		}
	}

	//setting a random number
	public static int randInt(int min, int max) {
		Random rand = new Random();
		int randomNum = rand.nextInt((max - min) + 1) + min;
		return randomNum;
	}

	//kFCV implementation
	public void KFoldCrossValidation() {

		ArrayList<Attributes> tempAttributeValueList = new ArrayList<Attributes>(attributeValueList);
		int rowPartition = (int) Math.ceil((attributeValueList.size() / partitionSize)) + 1;

		int randNum;

		kFoldPartition = new ArrayList<Attributes[]>(partitionSize);
		Attributes[] kFoldArray;
		while (tempAttributeValueList.size() != 0) {
			if (tempAttributeValueList.size() > rowPartition)
				kFoldArray = new Attributes[rowPartition];
			else
				kFoldArray = new Attributes[tempAttributeValueList.size()];

			for (int j = 0; j < rowPartition && tempAttributeValueList.size() != 0; j++) {

				randNum = randInt(0, tempAttributeValueList.size() - 1);
				kFoldArray[j] = tempAttributeValueList.get(randNum);
				tempAttributeValueList.remove(randNum);

			}
			kFoldPartition.add(kFoldArray);
		}
		System.out.println("Partitioned kfold");
		int i = 0;
		for (Attributes[] kf : kFoldPartition) {

			for (int j = 0; j < kf.length; j++) {
				System.out.println("Element " + j + " of " + i);
				System.out.print(kf[j]);
				System.out.println();
				System.out.print("----------------");
				System.out.println();

				//copying the partitions in text file.
				try (FileWriter fw = new FileWriter("Partition_" + i + ".txt", true);
						BufferedWriter bw = new BufferedWriter(fw);
						PrintWriter out = new PrintWriter(bw)) {
					out.println(kf[j].toString());
					out.println("---------------------------------------------");
				} catch (UnsupportedEncodingException e) {
					e.printStackTrace();
				} catch (FileNotFoundException e) {
					e.printStackTrace();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			System.out.print("\n");
			i++;
		}
		System.out.println("+++++++++++++++++");
	}

	//getting the testing partition randomly from the normalized k partitions
	public void SetTestingDataset() {
		int randNum = randInt(0, partitionSize-1);
		Attributes[] dataset = kFoldPartition.get(randNum);
		// Traversing the testing dataset
		int count = 0;
		System.out.println("----Training Dataset----");
		for (int j = 0; j < dataset.length; j++) {
			count++;
			System.out.println(count);
			System.out.print(dataset[j]);
			System.out.println();
			System.out.print("----------------");
			System.out.println();
		}
		kFoldPartition.remove(randNum); //remaining k-1 training partitions
		testingDataset = dataset;
	}

	public float[] getVarience(Attributes[] arr){
		
		if(arr.length < 1)
			return null;
					
		int ats = arr[0].getAttributeSize();
		float[] mean = new float[ats];
		float[] ans = new float[ats];
		
		int count = 0;
		
		for(int j=0;j<ats;j++)
		{
			mean[j] = 0;
			ans[j] = 0;
		}
		
		for(int i=0;i<arr.length;i++)
		{
			for(int j=0;j<arr[i].getAttributeSize();j++)
			{
				mean[j] += arr[i].getValue(j);
				count++;
			}
		}
		
		for(int j=0;j<ats;j++)
		{
			mean[j] = mean[j]/count;
		}
		
		for(int i=0;i<arr.length;i++)
		{
			for(int j=0;j<arr[i].getAttributeSize();j++)
			{
				float x = arr[i].getValue(j)-mean[j];
				ans[j] += (x*x);
			}
		}
		
		for(int j=0;j<ats;j++)
		{
			ans[j] = ans[j]/count;
		}
		
		return ans;
	}
	
	public void nearestDistance(int distCalculationChoice, int k) // change name
	{
		int eq=0,total=0;
		
		float[] varience = getVarience(testingDataset);
		
		for (int i=0;i<testingDataset.length;i++)
		{
			int trainingSize = attributeValueList.size() - testingDataset.length;
			ArrayList<DResult> trainingDataList = new ArrayList<DResult>(trainingSize); //this size is also contains the testing dataset
			ArrayList<DResult> tempTrainingDataList = null;
			
			DResult[] arr = new DResult[partitionSize-1];
			HashMap<String,CResult> di = new HashMap<String,CResult>();
			
			//getting 
			for(int j=0;j<partitionSize-1;j++) //for every partition element
			{
				tempTrainingDataList = getNearestClass(testingDataset[i],kFoldPartition.get(j),distCalculationChoice, varience);
				trainingDataList.addAll(tempTrainingDataList);
				tempTrainingDataList = null;
			}

			Collections.sort(trainingDataList);
//			System.out.println("========sorted list=========");
//			for(DResult d : trainingDataList){
//				System.out.println(d);
//				
//			}
			
			System.out.print("TestData - "+i+" - Real->"+testingDataset[i].className+" - Prediction->");
			
			for(int l = 0; l<k; l++){
				//System.out.println(d.classname +" -> "+d.distance);
				if(di.containsKey(trainingDataList.get(i).classname))
				{
					CResult cr=di.get(trainingDataList.get(i).classname);
					cr.count++;
				}
				else
				{
					CResult cr=new CResult();

					cr.cName = trainingDataList.get(i).classname;
					cr.count++;
					cr.min = trainingDataList.get(i).distance;

					di.put(cr.cName, cr);
				}
			}	
			CResult finalResult = null; 
			for (Map.Entry<String, CResult> entry : di.entrySet()) {
				String key = entry.getKey();
				CResult c = entry.getValue();

				//System.out.println(key +" -> "+ c.cName +" -> "+c.count);

				if(finalResult == null || finalResult.count < c.count)
					finalResult = c;
				else if(finalResult.count == c.count){
					if(c.min < finalResult.min)
						finalResult = c;
				}
			}
			
			System.out.println(finalResult.cName);
			if(finalResult.cName.equals(testingDataset[i].className))
			{
				eq++;
			}
			total++;
		}

		System.out.println("Accuracy = "+((100*eq)/total)+"%");
		System.out.println("---------------------------------------------------------------------");
	}

	private ArrayList<DResult> getNearestClass(Attributes testData, Attributes[] trainingPartition,int distCalculationChoice, float[] varience) {
		

		ArrayList<DResult> testingDataList = new ArrayList<DResult>();
		
		for(int i=0;i<trainingPartition.length;i++)
		{
			DResult temp=null;
			if(distCalculationChoice==1)
			{
				temp = testData.getPolynomialKernel(trainingPartition[i]);// change this
					
			}
			else if(distCalculationChoice==2)
			{
				temp = testData.getRadialKernel(trainingPartition[i], varience);
			}
			else
			{
				temp = testData.getEuclidian(trainingPartition[i]);
			}
			
			testingDataList.add(temp);

		}
		return testingDataList;
	}

}

