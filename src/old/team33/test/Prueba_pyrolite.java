package old.team33.test;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.RandomAccessFile;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;

import net.razorvine.pickle.PickleException;
import net.razorvine.pickle.Unpickler;;

public class Prueba_pyrolite {

    public static void main(String[] args) throws IOException {
    	final String pickleFilename = "w_out.p";
    	Path path = Paths.get(pickleFilename);
    	byte[] pickledata = Files.readAllBytes(path);
    	
    	RandomAccessFile f = new RandomAccessFile(pickleFilename, "r");
    	byte[] b = new byte[(int)f.length()];
    	f.read(b);
    	
    	InputStream stream = new FileInputStream(pickleFilename);

    	Unpickler unpickler = new Unpickler();
    	Object data = unpickler.load(stream);
    	
		Object result = unpickler.loads(pickledata);
		
		ArrayList<Double> w_out = (ArrayList<Double>) result;
    }

}
