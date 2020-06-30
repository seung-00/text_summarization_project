# https://github.com/pltrdy/rouge
# pip install rouge
from rouge import FilesRouge

if __name__ == "__main__":  
    path = "/Users/seungyoungoh/workspace/text_summarization_project/data/"

    files_rouge = FilesRouge()  
    scores_no_attention = files_rouge.get_scores(path+"seq2seq_no_attention/system/system.txt", path+"seq2seq_no_attention/reference/reference.txt", avg=True)
    print(scores_no_attention)

    scores_bahdanau_attention = files_rouge.get_scores(path+"seq2seq_bahdanau/system/system.txt", path+"seq2seq_bahdanau/reference/reference.txt", avg=True)
    print(scores_bahdanau_attention)
