package wm;

import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;
import org.apache.log4j.Logger;

import java.io.*;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * @program: lucene
 * @description: ${description}
 * @author: liushuaipeng
 * @create: 2019-12-14 12:45
 */
public class Utils {
    final static Logger LOGGER = Logger.getLogger(Utils.class);

    public static List<String[]> readCSV(String srcPath) {
        List<String[]> list = new ArrayList<String[]>();

        String charset = "utf-8";
        try (CSVReader csvReader = new CSVReaderBuilder(new BufferedReader(
                new InputStreamReader(new FileInputStream(new File(srcPath)),
                        charset))).build()) {
            Iterator<String[]> iterator = csvReader.iterator();
            while (iterator.hasNext()) {
                // Arrays.stream(iterator.next()).forEach(System.out::print);
                String str[] = iterator.next();
//                 LOGGER.info("id="+str[0]+"\t"+str[1]+"\t"+str[2]);
                // LOGGER.info();
                list.add(str);
            }
            csvReader.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        LOGGER.info(list.size());
        return list;
    }

//    public static String sign = "[[**##**]]";

//    public String keySentence(String para) {
//        String keySentence = "";
//
//        return keySentence;
//    }

    public static void write2File(String path, String str) {
        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(path));
            bw.write(str);
            bw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
