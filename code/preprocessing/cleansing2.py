import numpy as np 
import pandas as pd 

path= "/Users/seungyoungoh/workspace/text_summarization_project/data"
summary = pd.read_csv(path+'/news_summary.csv', encoding='iso-8859-1')
raw = pd.read_csv(path+'/news_summary_more.csv', encoding='iso-8859-1')
pre1 =  raw.iloc[:,0:2].copy()

pre2 = summary.iloc[:,0:6].copy()
pre2['text'] = pre2['author'].str.cat(pre2['date'].str.cat(pre2['read_more'].str.cat(pre2['text'].str.cat(pre2['ctext'], sep = " "), sep =" "),sep= " "), sep = " ")
pre = pd.DataFrame()
pre['text'] = pd.concat([pre1['text'], pre2['text']], ignore_index=True)
pre['summary'] = pd.concat([pre1['headlines'],pre2['headlines']],ignore_index = True)


news = pre
news = news.rename({'text':'src', 'summary':'smry'}, axis = 'columns')[['src','smry']]

news.to_csv(path+"/sample2.csv", mode='w')