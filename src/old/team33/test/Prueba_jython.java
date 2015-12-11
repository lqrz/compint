package old.team33.test;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.RandomAccessFile;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import net.razorvine.pickle.PickleException;
import net.razorvine.pickle.Unpickler;

import org.python.core.PyString;
import org.python.modules.cPickle;
import org.python.core.PyList;

public class Prueba_jython {

    public static void main(String[] args) throws IOException {
    	
    	final String pickleFilename = "w_out.p";

    	test_jython(pickleFilename);
    }
    
    private static void test_jython(String pickleFilename){
        File f = new File(pickleFilename);
        System.out.println(f.length());
        
        BufferedReader bufR;
        StringBuilder strBuilder = new StringBuilder();
        try {
            bufR = new BufferedReader(new FileReader(f));
            String aLine;
            while (null != (aLine = bufR.readLine())) {
                strBuilder.append(aLine).append("\n");
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        PyString pyStr = new PyString(strBuilder.toString());
        PyList pickle_java = (PyList) cPickle.loads(pyStr);
        System.out.println("Success");
    }
}
