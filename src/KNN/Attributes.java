package KNN;


import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
public class Attributes {
	
	static ArrayList<Float> min_attribute = null; //needed to normalize
	static ArrayList<Float> max_attribute = null; //needed to normalize
	static String[] attributeNames; //labels
	
	HashMap<String, Float> attributeValue = null; //main values

	public String className; 

	public Attributes(int size, String[] attributeNames) {

		Attributes.attributeNames = attributeNames;

		if (min_attribute == null || max_attribute == null) {
			min_attribute = new ArrayList<Float>(size-1);
			max_attribute = new ArrayList<Float>(size-1);

			for (int i = 0; i < size - 1; i++) {
				min_attribute.add(Float.MAX_VALUE);
				max_attribute.add(Float.MIN_VALUE);
			}
		}
		attributeValue = new HashMap<>(size);
	}

	public void parseString(String line) {
		String[] tokens = line.split(",");
		int i = 0;
		
		for (i = 0; i < tokens.length - 1; i++) {
			float val = Float.parseFloat(tokens[i]);
			if (min_attribute.get(i) > val) {
				min_attribute.set(i, val);
			}
			if (max_attribute.get(i) < val) {
				max_attribute.set(i, val);
			}
			attributeValue.put(attributeNames[i], val);
		}
		className = tokens[tokens.length - 1];
	}

	public void Normalize(){
		float newValue;
		for(int i= 0; i<attributeValue.size();i++){
			newValue = (attributeValue.get(attributeNames[i])-min_attribute.get(i))/(max_attribute.get(i)-min_attribute.get(i));
			attributeValue.replace(attributeNames[i],newValue);
		}				
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();

		Iterator it = attributeValue.entrySet().iterator();
		while (it.hasNext()) {
			Map.Entry pair = (Map.Entry) it.next();
			sb.append(pair.getKey() + " = " + pair.getValue() + "\n");
		}
		sb.append("Class = " + className);
		return sb.toString();
	}

	public static void printMinMax() {
		for (int i = 0; i < min_attribute.size(); i++) {
			System.out.println(attributeNames[i] + " -> Min=" + min_attribute.get(i) + ", Max->" + max_attribute.get(i));
		}
	}

	
	public DResult getEuclidian(Attributes attributes) {
		DResult ans = new DResult();
		float sum=0;
		
		for(int i=0;i<this.attributeValue.size();i++)
		{
			float x1 = this.attributeValue.get(attributeNames[i]); //test data element
			float x2 = attributes.attributeValue.get(attributeNames[i]);	//training data element
			sum += (x1-x2)*(x1-x2);
		}
		
		ans.distance = (float)Math.sqrt(sum);
		ans.classname = attributes.className;
		
		return ans;
	}
	
	public DResult getPolynomialKernel(Attributes attributes) {
		DResult ans = new DResult();
		float sum=0;
		
		for(int i=0;i<attributeValue.size();i++)
		{
			float x1 = attributeValue.get(attributeNames[i]);
			float x2 = attributes.attributeValue.get(attributeNames[i]);	
			//sum+= (x1*x1+1)-2*(x1*x2+1)+(x2*x2+1);
			//Kðx; yÞ¼ð1 þ hx; yiÞp
			sum += x1*x2;
		}
		ans.distance = (float)(1+sum)*(1+sum)*(1+sum);
		ans.classname = attributes.className;
		
		return ans;
	}

	


}



