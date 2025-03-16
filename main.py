#!/usr/bin/env python3
from speechRecognition.speechInput import recognizeSpeech
from roboticMovement.patternConstructor import createPatternFromObjects
from NER.entityRecognition import recogniseObjects
# gcloud auth application-default login        
# gcloud auth application-default set-quota-project capable-hash-452722-r5

if __name__ == "__main__":    
    request = recognizeSpeech()
    
    queue = recogniseObjects(request)
    
    print(queue)
    createPatternFromObjects(queue)
    
    