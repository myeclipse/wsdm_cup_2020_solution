package wm;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;

/**  
 * @author Liu Shuaipeng
 * @version V1.0  2015年8月30日
 * @since jdk1.6
 */
public class ConfUtil {
	public static Properties p =null;
	
	public static String getConfig(String str){
		p = new Properties();
		try {
			p.load(new BufferedInputStream(new FileInputStream("conf/conf.properties")));
//			p.load(new BufferedInputStream(new FileInputStream(ConfUtil.class.getClassLoader().getResource("conf.properties").getPath())));
			return p.getProperty(str);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return null;
	}
}
