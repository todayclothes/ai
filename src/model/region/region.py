from soynlp.tokenizer import LTokenizer
import joblib
import os

tokenizer = LTokenizer()
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'region_tokenizer_24.pkl')
tokenizer = joblib.load(file_path)

class Tokenizer():
        
    def __call__(self, data):
        list = []
        
        p = tokenizer.tokenize(data)
        
        for i in range(len(p)):
            if p[i].endswith('특별시') == True or p[i].endswith('광역시') == True:
                list.append(p[i][:-3])
                
            elif p[i].endswith('시') == True or p[i].endswith('군') == True or p[i].endswith('구') == True:
                list.append(p[i][:-1])
                    
            elif p[i].endswith('동') == True or p[i].endswith('면') == True or p[i].endswith('읍') == True or p[i].endswith('리') == True:
                list.append(p[i][:-1])
                
            elif p[i].endswith('역') == True:
                list.append(p[i][:-1])
                
            elif p[i].endswith('학교') == True:
                list.append(p[i][:-2])
               
            else:
                list.append(p[i])
                
        return ( " ".join( list ))