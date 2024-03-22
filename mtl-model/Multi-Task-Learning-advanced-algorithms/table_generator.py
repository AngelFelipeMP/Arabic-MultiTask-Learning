model
heads
data

from config 
INFO_DATA['DETOXIS']['metric'] = 'F1-score' -> 'f1_val'
INFO_DATA['EXIST']['metric'] = 'Accuracy' -> 'accuracy_val'
INFO_DATA['HatEval']['metric'] = 'F1-score' ('F1-macro') -> 'f1_macro_val'

"OSACT2022": "metric":"macro-f1" -> 'F1-macro' -> 'f1_val'
"HSARABIC": "metric":"f1_YES" -> 'F1-score' -> 'f1_macro_val'
"ArMI2021": "metric":"acc" -> 'accuracy' -> 'accuracy_val'