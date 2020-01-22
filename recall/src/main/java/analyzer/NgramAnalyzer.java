package analyzer;

//import org.apache.lucene.analysis.*;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.core.LowerCaseFilter;
import org.apache.lucene.analysis.core.StopFilter;
import org.apache.lucene.analysis.en.EnglishPossessiveFilter;
import org.apache.lucene.analysis.en.PorterStemFilter;
import org.apache.lucene.analysis.shingle.ShingleFilter;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.standard.StandardFilter;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;

import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;

/**
 * @program: wm20
 * @description: ${description}
 * @author: liushuaipeng
 * @create: 2020-01-03 00:01
 */
public class NgramAnalyzer extends Analyzer {
    private final CharArraySet stemExclusionSet;
    private final CharArraySet stopwords;

    public NgramAnalyzer() {
        stemExclusionSet = CharArraySet.EMPTY_SET;
        stopwords = getDefaultStopSet();
    }

    public static CharArraySet getDefaultStopSet() {
        return NgramAnalyzer.DefaultSetHolder.DEFAULT_STOP_SET;
    }

    @Override
    protected TokenStreamComponents createComponents(String arg0) {
//        final Tokenizer source = new NGramTokenizer(1, 2);
//        TokenStream result = new StandardFilter(source);
//
//        result = new EnglishPossessiveFilter(result);
//        result = new LowerCaseFilter(result);
//		    result = new StopFilter(result, this.stopwords);
//		    if(!stemExclusionSet.isEmpty())
//		      result = new SetKeywordMarkerFilter(result, stemExclusionSet);
//        result = new PorterStemFilter(result);
//        return new TokenStreamComponents(source, result);


//        result = new SetKeywordMarkerFilter(result, this.stemExclusionSet);
//        result = new GermanNormalizationFilter(result);
//        result = new NumberFilter(result);

        final Tokenizer source = new StandardTokenizer();
        TokenStream result = new StandardFilter(source);
        result = new EnglishPossessiveFilter(result);
        result = new PorterStemFilter(result);
        result = new StopFilter(result, this.stopwords);
        final ShingleFilter shingleFilter = new ShingleFilter(new LowerCaseFilter(result), 2);
        shingleFilter.setOutputUnigrams(true);
        return new TokenStreamComponents(source, shingleFilter);

    }

    private static class DefaultSetHolder {
        static final CharArraySet DEFAULT_STOP_SET;

        private DefaultSetHolder() {
        }

        static {
            DEFAULT_STOP_SET = StandardAnalyzer.STOP_WORDS_SET;
        }
    }


    public static List<String> generateNgrams(Analyzer analyzer, String str) throws IOException {
        List<String> result = new ArrayList<>();
        try {
            TokenStream stream = analyzer.tokenStream(null, new StringReader(str));
            stream.reset();
            while (stream.incrementToken()) {
                String nGram = stream.getAttribute(CharTermAttribute.class).toString();
                result.add(nGram);
                System.out.println("Generated N-gram = " + nGram);
            }
        } catch (IOException e) {
            System.out.println("IO Exception occured! " + e);
        }
        return result;
    }
//    static public void main(String[] args)
//            throws java.io.IOException {
//        Analyzer analyzer = new NgramAnalyzer();
//        List<String> nGrams = generateNgrams( analyzer, "sp600125 (anthra [1,9-cd] pyrazole-6 (2H)-one), a potent and selective JNK inhibitor, was obtained from EMD Biosciences. sp600125 inhibits JNKs (sp600125 acts as a reversible ATP-competitive JNK inhibitor) with an IC50 of 0.04–0.09 μm ([**##**]). sp600125 is selective for JNKs over ERKs and p38 (IC50 >10 μm). It was dissolved in 0.1% DMSO in saline. Based on the results demonstrating its ability to downregulate phosphorylation of JNKs downstream targets c-Jun, ATF2, and Elk1 a dose of 30 μm sp600125 was selected. The d-retro-inverso form of JNK-inhibitor (d-JNKI1) (H-Gly-d-Arg-d-Lys-d-Lys-d-Arg-d-Arg-d-Gln-d-Arg-d-Arg-d-Arg-d-Pro-d-Pro-d-Arg-d-Pro-d-Lys-d-Arg-d-Pro-d-Thr-d-Thr-d-Leu-d-Asn-d-Leu-d-Phe-d-Pro-d-Gln-d-Val-d-Pro-d-Arg-d-Ser-d-Gln-d-Asp-d-Thr-NH2) was obtained from GL Biochem. The d-JNKI1 was initially solubilized in DMSO to make a 6.37 mm stock solution. The same stock solution of d-JNKI1 was further diluted in saline to obtain a final concentration of 4 or 8 μm. Anisomycin (Sigma) was used as a potent activator of JNKs. It was first dissolved in 1 m HCl the pH subsequently adjusted to 7.0 with 1 m NaOH. The final concentration of 100 μg/μl was obtained by adding 0.9% saline . Human or rat CRF (h/rCRF) and the CRF2 antagonist antisauvagine-30 (aSvg-30) were synthesized as described previously . The peptides were initially dissolved in 10 mm acetic acid and diluted with twofold concentrated sterile artificial CSF (aCSF). The final pH of the peptide solutions was 7.4. The final concentrations of h/rCRF and aSvg-30 were selected on the basis of our previous experiments" );
//
//        for ( String nGram : nGrams ) {
//            System.out.println( nGram );
//        }
//    }

}
