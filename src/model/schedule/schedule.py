from soynlp.tokenizer import LTokenizer
import joblib
import os

tokenizer = LTokenizer()
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'schedule_tokenizer_5.pkl')
tokenizer = joblib.load(file_path)

class schedule_Tokenizer():
        
    def __call__(self, data):
        list = []
        
        p = tokenizer.tokenize(data)
        
        for i in range(len(p)):
                list.append(p[i])
                
        return ( " ".join( list ))