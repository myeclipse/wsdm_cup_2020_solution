import time
import  numpy as np
import nltk
# nltk.download('punkt')
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import re


def tokenize(sentence):
    '''
        去除多余空白、分词、词性标注
    '''
    sentence = re.sub(r'\s+', ' ', sentence)
    token_words = word_tokenize(sentence)  # 输入的是列表
    token_words = pos_tag(token_words)
    return token_words


def stem(token_words):
    '''
        词形归一化
    '''
    wordnet_lematizer = WordNetLemmatizer()  # 单词转换原型
    words_lematizer = []
    for word, tag in token_words:
        if tag.startswith('NN'):
            word_lematizer = wordnet_lematizer.lemmatize(word, pos='n')  # n代表名词
        elif tag.startswith('VB'):
            word_lematizer = wordnet_lematizer.lemmatize(word, pos='v')  # v代表动词
        elif tag.startswith('JJ'):
            word_lematizer = wordnet_lematizer.lemmatize(word, pos='a')  # a代表形容词
        elif tag.startswith('R'):
            word_lematizer = wordnet_lematizer.lemmatize(word, pos='r')  # r代表代词
        else:
            word_lematizer = wordnet_lematizer.lemmatize(word)
        words_lematizer.append(word_lematizer)
    return words_lematizer


sr = stopwords.words('english')


def delete_stopwords(token_words):
    '''
        去停用词
    '''
    cleaned_words = [word for word in token_words if word not in sr]
    return cleaned_words


def is_number(s):
    '''
        判断字符串是否为数字
    '''
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


characters = [' ', ',', '.', 'DBSCAN', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '-', '...',
              '^', '{', '}']


def delete_characters(token_words):
    '''
        去除特殊字符、数字
    '''
    words_list = [word for word in token_words if word not in characters and not is_number(word)]
    return words_list


def to_lower(token_words):
    '''
        统一为小写
    '''
    words_lists = [x.lower() for x in token_words]
    return words_lists


def pre_process(text):
    '''
        文本预处理
    '''
    token_words = tokenize(text)
    token_words = stem(token_words)
    token_words = delete_stopwords(token_words)
    token_words = delete_characters(token_words)
    token_words = to_lower(token_words)
    return token_words





#多进程函数
import  pandas as pd
from multiprocessing import cpu_count,Pool
'多进程处理'
def pool_extract(data, f ,vec_model,chunk_size, worker=4):
    cpu_worker = cpu_count()
    print('cpu core:{}'.format(cpu_worker))
    if worker == -1 or worker > cpu_worker:
        worker = cpu_worker
    print('use cpu:{}'.format(worker))
    t1 = time.time()
    len_data = len(data)
    start = 0
    end = 0
    p = Pool(worker)
    res = []  # 保存的每个进程的返回值
    while end < len_data:
        end = start + chunk_size
        if end > len_data:
            end = len_data
        rslt = p.apply_async(f, (data[start:end],vec_model))
        start = end
        res.append(rslt)
    p.close()
    p.join()
    t2 = time.time()
    print((t2 - t1)/60)
    results = pd.concat([i.get() for i in res], axis=0, ignore_index=True)
    return results


#####reduce mem
import datetime
def pandas_reduce_mem_usage(df):
    start_mem=df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    starttime = datetime.datetime.now()
    for col in df.columns:
        col_type=df[col].dtype   #每一列的类型
        if col_type !=object:    #不是object类型
            c_min=df[col].min()
            c_max=df[col].max()
            # print('{} column dtype is {} and begin convert to others'.format(col,col_type))
            if str(col_type)[:3]=='int':
                #是有符号整数
                if c_min<0:
                    if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min >= np.iinfo(np.uint8).min and c_max<=np.iinfo(np.uint8).max:
                        df[col]=df[col].astype(np.uint8)
                    elif c_min >= np.iinfo(np.uint16).min and c_max <= np.iinfo(np.uint16).max:
                        df[col] = df[col].astype(np.uint16)
                    elif c_min >= np.iinfo(np.uint32).min and c_max <= np.iinfo(np.uint32).max:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
            #浮点数
            else:
                if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
            # print('\t\tcolumn dtype is {}'.format(df[col].dtype))

        #是object类型，比如str
        else:
            # print('\t\tcolumns dtype is object and will convert to category')
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024 ** 2
    endtime = datetime.datetime.now()
    print('consume times: {:.4f}'.format((endtime - starttime).seconds))
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

if __name__ == '__main__':
    text = 'This experiment was conducted to determine whether feeding meal and hulls derived from genetically modified soybeans to dairy cows affected production measures and sensory qualities of milk. The soybeans were genetically modified (Event DAS-444Ø6-6) to be resistant to multiple herbicides. Twenty-six Holstein cows (13/treatment) were fed a diet that contained meal and hulls derived from transgenic soybeans or a diet that contained meal and hulls from a nontransgenic near-isoline variety. Soybean products comprised approximately 21% of the diet dry matter, and diets were formulated to be nearly identical in crude protein, neutral detergent fiber, energy, and minerals and vitamins. The experimental design was a replicated 2×2 Latin square with a 28-d feeding period. Dry matter intake (21.3 vs. 21.4kg/d), milk yield (29.3 vs. 29.4kg/d), milk fat (3.70 vs. 3.68%), and milk protein (3.10 vs. 3.12%) did not differ between cows fed control or transgenic soybean products, respectively. Milk fatty acid profile was virtually identical between treatments. Somatic cell count was significantly lower for cows fed transgenic soybean products, but the difference was biologically trivial. Milk was collected from all cows in period 1 on d 0 (before treatment), 14, and 28 for sensory evaluation. On samples from all days (including d 0) judges could discriminate between treatments for perceived appearance of the milk. The presence of this difference at d 0 indicated that it was likely not a treatment effect but rather an initial bias in the cow population. No treatment differences were found for preference or acceptance of the milk. Overall, feeding soybean meal and hulls derived from this genetically modified soybean had essentially no effects on production or milk acceptance when fed to dairy cows. '
    text = 'Pyrvinium is a drug approved by the FDA and identified as a Wnt inhibitor by inhibiting Axin degradation and stabilizing 尾-catenin, which can increase Ki67+ cardiomyocytes in the peri-infarct area and alleviate cardiac remodeling in a mouse model of MI . UM206 is a peptide with a high homology to Wnt-3a/5a, and acts as an antagonist for Frizzled proteins to inhibit Wnt signaling pathway transduction. UM206 could reduce infarct size, increase the numbers of capillaries, decrease myofibroblasts in infarct area of post-MI heart, and ultimately suppress the development of heart failure . ICG-001, which specifically inhibits the interaction between 尾-catenin and CBP in the Wnt canonical signaling pathway, can promote the differentiation of epicardial progenitors, thereby contributing to myocardial regeneration and improving cardiac function in a rat model of MI . Small molecules invaliding Porcupine have been further studied, such as WNT-974, GNF-6231 and CGX-1321. WNT-974 decreases fibrosis in post-MI heart, with a mechanism of preventing collagen production in cardiomyocytes by blocking secretion of Wnt-3, a pro-fibrotic agonist, from cardiac fibroblasts and its signaling to cardiomyocytes . The phosphorylation of DVL protein is decreased in both the canonical and non-canonical Wnt signaling pathways by WNT-974 administration . GNF-6231 prevents adverse cardiac remodeling in a mouse model of MI by inhibiting the proliferation of interstitial cells, increasing the proliferation of Sca1+ cardiac progenitors and reducing the apoptosis of cardiomyocytes [[**##**]]. Similarly, we demonstrate that CGX-1321, which has also been applied in a phase I clinical trial to treat solid tumors ({"type":"clinical-trial","attrs":{"text":"NCT02675946","term_id":"NCT02675946"}}NCT02675946), inhibits both canonical and non-canonical Wnt signaling pathways in post-MI heart. CGX-1321 promotes cardiac function by reducing fibrosis and stimulating cardiomyocyte proliferation-mediated cardiac regeneration in a Hippo/YAP-independent manner . These reports implicate that Wnt pathway inhibitors are a class of potential drugs for treating MI through complex mechanisms, including reducing cardiomyocyte death, increasing angiogenesis, suppressing fibrosis and stimulating cardiac regeneration.'
    token_words = tokenize(text)
    print(token_words)
    token_words = stem(token_words)  # 单词原型
    token_words = delete_stopwords(token_words)  # 去停
    token_words = delete_characters(token_words)
    token_words = to_lower(token_words)
    print(token_words)

