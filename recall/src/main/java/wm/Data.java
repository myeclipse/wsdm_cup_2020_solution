package wm;

/**
 * @program: lucene
 * @description: ${description}
 * @author: liushuaipeng
 * @create: 2019-12-16 15:14
 */
public class Data {
    StringBuffer sbfPaperId = new StringBuffer();
    StringBuffer sbfScore = new StringBuffer();
    int recallNum = 0;
    int totalNum = 0;
    String threadName;
    double top3Score = 0.0;

    public Data() {
    }

    @Override
    public String toString() {
        return "Data{" +
                "recallNum=" + recallNum +
                ", totalNum=" + totalNum +
                ", threadName='" + threadName + '\'' +
                ", top3Score=" + top3Score +
                '}';
    }

    public StringBuffer getSbfScore() {
        return sbfScore;
    }

    public void setSbfScore(StringBuffer sbfScore) {
        this.sbfScore = sbfScore;
    }

    public double getTop3Score() {
        return top3Score;
    }

    public void setTop3Score(double top3Score) {
        this.top3Score = top3Score;
    }

    public StringBuffer getSbfPaperId() {
        return sbfPaperId;
    }

    public void setSbfPaperId(StringBuffer sbfPaperId) {
        this.sbfPaperId = sbfPaperId;
    }

    public int getRecallNum() {
        return recallNum;
    }

    public void setRecallNum(int recallNum) {
        this.recallNum = recallNum;
    }

    public int getTotalNum() {
        return totalNum;
    }

    public void setTotalNum(int totalNum) {
        this.totalNum = totalNum;
    }

    public String getThreadName() {
        return threadName;
    }

    public void setThreadName(String threadName) {
        this.threadName = threadName;
    }

}