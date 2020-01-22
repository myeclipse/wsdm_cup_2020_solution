import analyzer.NgramAnalyzer;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;

/**
 * @program: wm20
 * @description: ${description}
 * @author: liushuaipeng
 * @create: 2020-01-04 13:49
 */
public class Test {

    public static void parserTest1() {
        Analyzer analyzer = analyzer = new NgramAnalyzer();
//        String str = "sp600125 (anthra [1,9-cd] pyrazole-6 (2H)-one), a potent and selective JNK inhibitor, was obtained from EMD Biosciences. sp600125 inhibits JNKs (sp600125 acts as a reversible ATP-competitive JNK inhibitor) with an IC50 of 0.04–0.09 μm . sp600125 is selective for JNKs over ERKs and p38 (IC50 >10 μm). It was dissolved in 0.1% DMSO in saline. Based on the results demonstrating its ability to downregulate phosphorylation of JNKs downstream targets c-Jun, ATF2, and Elk1 a dose of 30 μm sp600125 was selected. The d-retro-inverso form of JNK-inhibitor (d-JNKI1) (H-Gly-d-Arg-d-Lys-d-Lys-d-Arg-d-Arg-d-Gln-d-Arg-d-Arg-d-Arg-d-Pro-d-Pro-d-Arg-d-Pro-d-Lys-d-Arg-d-Pro-d-Thr-d-Thr-d-Leu-d-Asn-d-Leu-d-Phe-d-Pro-d-Gln-d-Val-d-Pro-d-Arg-d-Ser-d-Gln-d-Asp-d-Thr-NH2) was obtained from GL Biochem. The d-JNKI1 was initially solubilized in DMSO to make a 6.37 mm stock solution. The same stock solution of d-JNKI1 was further diluted in saline to obtain a final concentration of 4 or 8 μm. Anisomycin (Sigma) was used as a potent activator of JNKs. It was first dissolved in 1 m HCl the pH subsequently adjusted to 7.0 with 1 m NaOH. The final concentration of 100 μg/μl was obtained by adding 0.9% saline ([**##**]). Human or rat CRF (h/rCRF) and the CRF2 antagonist antisauvagine-30 (aSvg-30) were synthesized as described previously . The peptides were initially dissolved in 10 mm acetic acid and diluted with twofold concentrated sterile artificial CSF (aCSF). The final pH of the peptide solutions was 7.4. The final concentrations of h/rCRF and aSvg-30 were selected on the basis of our previous experiments .";
        String str = "The half-life in circulation of hPP is short at 7 min [[**##**]] and previous studies on its physiological effects have overcome this problem by using prolonged, continuous intravenous infusions. As with other peptide-based hormones such as insulin, the simplest and most practical method of hPP administration would be subcutaneous injection. However, subcutaneous hPP would not be expected to deliver sustained concentrations for long. To overcome the short half-life problem, we have developed peptidase-resistant analogues of hPP to possess the following properties: longer half-lives, selective Y4 receptor binding activity and powerful appetite-suppressive properties in validated pre-clinical models of obesity (unpublished data). One of these, PP 1420, has an amino acid sequence that is similar to that of hPP, with one additional amino acid compared with the hPP sequence (a glycine residue located at position 0) and substitutions of five other residues of PP within the 37-residue peptide. The peptide is C-terminally amidated and contains only standard L-amino acids. The chemical name for PP 1420 is L-glycyl-L-alanyl-L-prolyl-L-leucyl-L-glutamyl-L-prolyl-L-valyl-L-tyrosyl-L-prolyl-L-glycyl-L-aspargyl-L-asparginyl-L-alanyl-L-threonyl-L-prolyl-L-glutamyl-L-glutaminyl-L-lysyl-L-alanyl-L-lysyl-L-tyrosyl-L-alanyl-L-alanyl-L-glutamyl-L-leucyl-L-arginyl-L-arginyl-L-tyrosyl-L-isoleucyl-L-aspargyl-L-arginyl-L-leucyl-L-threonyl-L-arginyl-L-prolyl-L-arginyl-L-tyrosinamide, hydrochloride salt.";
        String queryText = QueryParser.escape(str.replaceAll("[\r\n+\\(\\)\\[\\]\\*\\{\\\\/#_-]", " "));
        System.out.println(queryText);
    }

    public static void parserTest2() throws ParseException {
        Analyzer analyzer = new EnglishAnalyzer();
        QueryParser parser = new QueryParser("", analyzer);
        String text = "";
        String ret = parser.parse(QueryParser.escape(text)).toString();
        System.out.println(ret);
    }

    public static void main(String args[]) throws Exception {
//    Test.parserTest2();
        String s="                  ";
        s = s.trim();
        System.out.println(s.length());
    }
}
