package wm;

import analyzer.NgramAnalyzer;
import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.*;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.*;
import org.apache.lucene.search.similarities.*;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.RAMDirectory;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.regex.Pattern;

/**
 * @program: lucene
 * @description: ${description}
 * @author: liushuaipeng
 * @create: 2019-12-14 12:42
 */


public class LuceneIndex implements Runnable {
    final static Logger LOGGER = Logger.getLogger(LuceneIndex.class);
    static boolean create = true;

    Similarity sim = null;

    //    Analyzer analyzer = new StandardAnalyzer();
//    Analyzer analyzer = new EnglishAnalyzer();
//  Analyzer analyzer = new NgramAnalyzer();

    Analyzer analyzer = null;
    Directory dir = null;
    List<String[]> list = null;
    Data data = null;

    String indexType;
    int topn;
    String simType;
    String createIndexType;

    public LuceneIndex(Directory dir, List<String[]> list, Data data, String indexType, int topn, String simType, String createIndexType, String analyzer) {
        this(indexType, topn, simType, createIndexType, analyzer);
        this.dir = dir;
        this.list = list;
        this.data = data;
    }

    public LuceneIndex(String indexType, int topn, String simType, String createIndexType, String analyzerStr) {
        this.indexType = indexType;
        this.topn = topn;
        this.simType = simType;
        this.createIndexType = createIndexType;

        if ("ngram".equals(analyzerStr))
            this.analyzer = new NgramAnalyzer();
        else if ("english".equals(analyzerStr))
            this.analyzer = new EnglishAnalyzer();
        else if ("standard".equals(analyzerStr))
            this.analyzer = new StandardAnalyzer();


        if (simType.startsWith("bm25")) {
            String[] bm25Args = simType.split("-");
            if (bm25Args.length == 3)
               sim = new BM25Similarity(Float.parseFloat(bm25Args[1]), Float.parseFloat(bm25Args[2]));
            else
                sim = new BM25Similarity(1.2F, 0.75F);
        } else if (simType.equals("tfidf"))
            sim = new ClassicSimilarity();
        else if (simType.equals("lmd"))
            sim = new LMDirichletSimilarity();
        else if (simType.equals("F3EXP"))
            sim = new AxiomaticF3EXP(0.25F, 1, 0.35F);
        else if (simType.equals("DFI"))
            sim = new DFISimilarity(new IndependenceChiSquared());

        else if (simType.equals("F1EXP"))
            sim = new AxiomaticF1EXP();
        else if (simType.equals("F2EXP"))
            sim = new AxiomaticF2EXP();
        else if (simType.equals("F1LOG"))
            sim = new AxiomaticF1LOG();
        else if (simType.equals("F2LOG"))
            sim = new AxiomaticF2LOG();
        else if (simType.equals("F3LOG"))
            sim = new AxiomaticF3LOG(0.5F, 10);
        else if (simType.equals("boolean"))
            sim = new BooleanSimilarity();
        else if (simType.equals("LMJ"))
            sim = new LMJelinekMercerSimilarity(0.2F);
        else if (simType.equals("DFR"))
            sim = new DFRSimilarity(new BasicModelBE(), new AfterEffectB(), new NormalizationH1());
        else if (simType.equals("IB"))
            sim = new IBSimilarity(new DistributionLL(), new LambdaDF(), new NormalizationH3());


        this.dir = new RAMDirectory();
        LOGGER.info("topN=" + topn);
    }


    public Directory getDir() {
        return this.dir;
    }

    /**
     * Indexes a single document
     */
    void indexADoc(IndexWriter writer, String paperID, String content, String journal)
            throws IOException {
        if (paperID.length() != 24) {
            LOGGER.info(paperID);
            return;
        }
        if (journal.equals("no-content") && indexType.equals("other"))
            return;
        if (!journal.equals("no-content") && indexType.equals("no_content"))
            return;

        Document doc = new Document();
        Field pathField = new StringField("paperID", paperID, Field.Store.YES);//不分词
        doc.add(pathField);
//        doc.add(new StringField("journal", journal, Field.Store.YES));//不分词
        doc.add(new TextField("content", (null == content || content.length() == 0) ? "kongchuan" : content, Field.Store.YES));//分词
        if (writer.getConfig().getOpenMode() == IndexWriterConfig.OpenMode.CREATE) {
            writer.addDocument(doc);
        } else {
            LOGGER.info("updating " + paperID);
            writer.updateDocument(new Term("content", content), doc);
        }
    }

    void indexAll(String filePath) throws IOException {
        IndexWriterConfig iwc = new IndexWriterConfig(analyzer);
        iwc.setSimilarity(sim);
        if (create) {
            iwc.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
        } else {
            iwc.setOpenMode(IndexWriterConfig.OpenMode.CREATE_OR_APPEND);
        }
        IndexWriter writer = new IndexWriter(dir, iwc);
        List<String[]> paperList = Utils.readCSV(filePath);
        //0:abstract
        //1:journal
        //2:keywords
        //3:paper_id
        //4:titile
        //5:year
        LOGGER.info("论文数量=" + paperList.size());
        for (String[] paper : paperList) {

            if (paper[0].trim().equals("abstract"))
                continue;

            String content = null;

            if (createIndexType.equals("all"))
                content = String.format("%s %s %s %s %s", paper[0], paper[1], paper[2], paper[4], paper[5]).replaceAll("[\n\r]", "").trim().toLowerCase();
            else if (createIndexType.equals("title"))
                content = String.format("%s ", paper[4]).replaceAll("[\n\r]", "").toLowerCase();
            else if (createIndexType.equals("abstract"))
                content = String.format("%s ", paper[0]).replaceAll("[\n\r]", "").toLowerCase();
            else if (createIndexType.equals("keywords"))
                content = String.format("%s ", paper[2]).replaceAll("[\n\r]", "").toLowerCase();

            this.indexADoc(
                    writer,
                    paper[3],
                    content,
                    paper[1]
            );
        }
        writer.commit();
    }

    @Override
    public void run() {
        Pattern pattern = Pattern.compile("[\r\t\n+\\(\\)\\[\\]\\*\\{\\\\/#_-]");
        DecimalFormat fmt = new DecimalFormat("##0.0");
        BooleanQuery.setMaxClauseCount(1000000);
        StringBuffer sbfPaperID = new StringBuffer();
        StringBuffer sbfScore = new StringBuffer();
        IndexReader reader = null;
        String queryText = "";

        Analyzer analyzer = null;
        if (this.analyzer instanceof StandardAnalyzer)
            analyzer = new StandardAnalyzer();
        else if (this.analyzer instanceof EnglishAnalyzer)
            analyzer = new EnglishAnalyzer();
        else if (this.analyzer instanceof NgramAnalyzer)
            analyzer = new NgramAnalyzer();

        try {
            reader = DirectoryReader.open(this.dir);
            IndexSearcher searcher = new IndexSearcher(reader);
            searcher.setSimilarity(sim);
            QueryParser parser = new QueryParser("content", analyzer);
            int matchNum = 0;
            double top3Score = 0.0;

            for (int index = 0; index < this.list.size(); index++) {
                LOGGER.info(Thread.currentThread().getName() + " 进度" + index + "/" + this.list.size());

                String[] aPress = this.list.get(index);
                LOGGER.info(aPress.length);
                String pressID = aPress[0].trim();

                if (pressID.contains("description_id"))
                    continue;

                String text = null;
                if (aPress.length == 3)
                    text = aPress[2];
                else
                    text = aPress[1];

//                queryText = QueryParser.escape(text).trim().replaceAll("[\r\n+\\(\\[\\]\\*\\{_-]", " ").toLowerCase();
//                queryText = QueryParser.escape(text.replaceAll("[\r\n+\\(\\)\\[\\]\\*\\{\\\\/#_-]", " ").toLowerCase());
                queryText = QueryParser.escape(pattern.matcher(text).replaceAll(" ").toLowerCase().trim());
                if (queryText.length() == 0)
                    queryText = "kongzifu";

                Query query = parser.parse(queryText);

                TopDocs results = searcher.search(query, topn);

                ScoreDoc[] hits = results.scoreDocs;

                int numTotalHits = results.totalHits;
//                LOGGER.info(numTotalHits + " total matching documents");

                sbfPaperID.append(pressID).append(",");
                sbfScore.append(pressID).append(",");

                int rightIdx = 0;
                int min = numTotalHits < topn ? numTotalHits : topn;

                for (int i = 0; i < min; i++) {
                    Document doc = searcher.doc(hits[i].doc);
                    double score = hits[i].score;
                    score = Double.parseDouble(fmt.format(score));
                    String pid = doc.get("paperID");

                    if (pid.equals(aPress[1])) {
                        matchNum += 1;
                        rightIdx = i + 1;
                        if (i < 3)
                            top3Score += 1.0 / (i + 1);
                    }

                    if (i == topn - 1) {
                        sbfPaperID.append(pid).append("\n");
                        sbfScore.append(score).append(",").append(rightIdx).append("\n");
                    } else {
                        sbfPaperID.append(pid).append(",");
                        sbfScore.append(score).append(",");
                    }
                }


                for (int i = topn - min; i > 0; i--) {
                    if (i == 1) {
                        sbfPaperID.append("5c0f7eebda562944ac8215e7").append("\n");
                        sbfScore.append(-1).append(",").append(rightIdx).append("\n");
                    } else {
                        sbfPaperID.append("5c0f7eebda562944ac8215e7").append(",");
                        sbfScore.append(-1).append(",");
                    }
                }
            }


            this.data.setRecallNum(matchNum);
            this.data.setTotalNum(this.list.size());
            this.data.setThreadName(Thread.currentThread().getName());
            this.data.setSbfPaperId(sbfPaperID);
            this.data.setTop3Score(top3Score);
            this.data.setSbfScore(sbfScore);
            LOGGER.info(Thread.currentThread().getName() + " match_num=" + matchNum + " percent=" + 1.0 * matchNum / this.list.size() + "\ttop3Score=" + top3Score / this.list.size());
        } catch (Exception e) {
            e.printStackTrace();
            LOGGER.error("queryText=" + queryText + "length=" + queryText.length());
            LOGGER.error("error info:", e);
        }
    }

    public static void main(String[] args) throws Exception {
        PropertyConfigurator.configureAndWatch("conf/log4j.properties");

        String candidate = ConfUtil.getConfig("candidate_paper_path");
        int topn = Integer.parseInt(ConfUtil.getConfig("topn"));

        String indexType = ConfUtil.getConfig("index_type");
        String simType = ConfUtil.getConfig("sim");
        String createIndexType = ConfUtil.getConfig("createIndexType");
        String des = ConfUtil.getConfig("description_path");
        int threadNum = Integer.parseInt(ConfUtil.getConfig("threadNum"));
        String analyzerStr = ConfUtil.getConfig("analyzer");


        long start = System.currentTimeMillis();

        LuceneIndex li = new LuceneIndex(indexType, topn, simType, createIndexType, analyzerStr);
        LOGGER.info(li.sim.toString());
        LOGGER.info(candidate);
        li.indexAll(candidate);

        LOGGER.info(des);
        List<String[]> pressList = Utils
                .readCSV(des);

        LOGGER.info(pressList.size());

        LOGGER.info("设置线程数=" + threadNum);
        int segNum;
        if (pressList.size() % threadNum == 0)
            segNum = pressList.size() / threadNum;
        else
            segNum = pressList.size() / threadNum + 1;
        ExecutorService exec = Executors.newCachedThreadPool();

        List<Data> dataList = new ArrayList(threadNum);
        int t = 0;
        for (int i = 0; i < threadNum; i++) {
            Data data = new Data();
            dataList.add(data);
            int startIdx = i * segNum;
            int endIdx = (i + 1) * segNum > pressList.size() ? pressList.size() : (i + 1) * segNum;
            LOGGER.info(Thread.currentThread().getName() + i + " startIdx=" + startIdx + " endIdx=" + endIdx);
            List<String[]> sub_list = pressList.subList(startIdx, endIdx);
            t += sub_list.size();
            LuceneIndex thread_li = new LuceneIndex(li.getDir(), sub_list, data, indexType, topn, simType, createIndexType, analyzerStr);
            exec.submit(thread_li);
        }
        LOGGER.info(t);
        exec.shutdown();
        while (true) {
            if (exec.isTerminated()) {
                LOGGER.info("所有的子线程都结束了！");
                break;
            }
            Thread.sleep(1000);
        }
        final long end = System.currentTimeMillis();
        System.out.println("耗时(s)：" + (end - start) / 1000);


        int total_num = 0;
        int recall_num = 0;
        StringBuffer sbfPaperID = new StringBuffer();
        StringBuffer sbfScore = new StringBuffer();
        double top3Score = 0.0;

//        #统计recall
        for (Data data : dataList) {
            LOGGER.info(data.toString());
            total_num += data.getTotalNum();
            recall_num += data.getRecallNum();
            sbfPaperID.append(data.getSbfPaperId());
            sbfScore.append(data.getSbfScore());
            top3Score += data.getTop3Score();
        }

        String strs[] = des.split("/");

        String recallFilePath = String.format("./recall_%s_%s_%s_%d_%s", analyzerStr, simType, createIndexType, topn, strs[strs.length - 1]);
        Utils.write2File(recallFilePath, sbfPaperID.toString());

        String scoreFilePath = String.format("./score_%s_%s_%s_%d_%s", analyzerStr, simType, createIndexType, topn, strs[strs.length - 1]);
        Utils.write2File(scoreFilePath, sbfScore.toString());

        LOGGER.error("analyzerStr=" + analyzerStr + "\tsimType=" + simType + "\ttopN=" + li.topn + "\trecallNum:" + recall_num + "\ttotal:" + total_num + "\trecall=" + 1.0 * recall_num / total_num + "\ttop3Score=" + top3Score / total_num);

    }
}
