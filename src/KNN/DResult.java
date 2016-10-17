package KNN;

public  class DResult implements Comparable<DResult>{
	float distance;
	String classname;
	
	
	public int compareTo(DResult o) {
		if(distance > o.distance)
			return 1;
		else if(distance < o.distance)
			return -1;
		else
			return 0;
	}
	
	@Override
	public String toString() {
		return new StringBuffer().append(classname).append(" -> ").append(distance).toString();
	}
	
}
