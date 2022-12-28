import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
# !pip install --upgrade category_encoders
from category_encoders import OrdinalEncoder
from sklearn.metrics import accuracy_score

text_us=pd.read_csv(r"C:\Users\Avinash\Desktop\Assingments\AML\us.txt", header=None)
text_us.head()

text_us.columns=["Name"]
text_us['Language']='USA'
text_us.head()

# Importing the America dataset and adding Language as USA for that data set as a data in column of Language by creating it.
text_japan=pd.read_csv(r"C:\Users\Avinash\Desktop\Assingments\AML\japan.txt", header=None)
text_japan.head()

text_japan.columns=["Name"]
text_japan['Language']='Japanese'
text_japan.head()
# Importing the Japanese dataset and adding Language as Japanese for that data set as a data in column of Language by creating it.

# %%
text_arabic=pd.read_csv(r"C:\Users\Avinash\Desktop\Assingments\AML\arabic.txt", header=None)
text_arabic.head()

text_arabic.columns=["Name"]
text_arabic['Language']='Arabic'
text_arabic.head()

# Importing the Arabic dataset and adding Language as Arabic for that data set as a data in column of Language by creating it.
text_greek=pd.read_csv(r"C:\Users\Avinash\Desktop\Assingments\AML\greek.txt", header=None)
text_greek.head()

text_greek.columns=["Name"]
text_greek['Language']='Greek'
text_greek.head()


# Importing the Greek dataset and adding Language as Greek for that data set as a data in column of Language by creating it.

text_total= pd.concat([text_us, text_arabic, text_greek, text_japan], axis=0, ignore_index=True)
print(text_total.shape)
text_total.head()

# Concatenating the four(American, Japanese, Arabic and Greek data set in row wise by making Name and Language as name of the Columns in the final final dataset and making it ready to do preprocessing in the next steps

vectorize_transformation=CountVectorizer().fit(text_total['Name'])
text_transformed=vectorize_transformation.transform(text_total['Name']).toarray()
print(text_transformed)


# Using CountVectorizer to transform the data in the Name column having different names in different languages and converting it into matrix based on the frequency of the words that occur in the Name column in 4 different languages so that the data can be used to train the Naice Bayes Machine Learning Model to classify the testing data set

language_mapping=[{'col': 'Language', 'mapping': {'USA':0, 'Japanese':1,'Arabic':2,'Greek':3}}]
encoder_language=OrdinalEncoder(mapping=language_mapping)
text_total=encoder_language.fit_transform(text_total)


# Using OrdinalEncoder to map the Language name present in the Language column to integers from 0 to 3 so that they can be used to train the model



y=text_total['Language']
print(y)
x_train,x_test,y_train,y_test=train_test_split(text_transformed, y, test_size = 0.3, shuffle=True)

# Splitting the datset after preprocessing into testing and training datasets in 70 and 30 percentages respectively

samples_number,feature_number=x_train.shape
class_probability=np.zeros(4)
count,uniq_classes=np.unique(y_train,return_counts=True)
#print(uniq_classes)
i=0
while i<4:
  class_probability[i]=uniq_classes[i]/samples_number
  i+=1
print(class_probability)



# Finding the probability of each data in Language column by Bayes multivariate  probability formula

class Naive_Bayes:     
    def model_train(self, x, y): 
        total_samples, num_of_feat = x_train.shape      #finding the total number of sampled and features
        diff_classes_1d = np.unique(y)                    #finding the uniques classes based on the req output
        num_of_cla = diff_classes_1d.shape[0]         #findinf the number of classses (4 here)
        #print(num_of_cla)
        lst=[]
        for i in range(num_of_feat):
            val = np.unique(x[:,i])              #appending the uniques values in the training data set present in the column of Name (matrix) based on thefeatures
            lst.append(val)
        #print(lst)
            
        self.total_y2d =np.zeros((num_of_cla, num_of_feat))
        self.total_y1d =np.zeros((num_of_cla))
        for i in diff_classes_1d:                                         #looping through number of classes 
            start_index =np.argwhere(y.to_numpy()==i).flatten()
            sum_of_col =[]
            for j in range(num_of_feat):
                sum_of_col.append(np.sum(x[start_index,j]))
            #print(sum_of_col)  
            self.total_y2d[i] =sum_of_col
            self.total_y1d[i] =np.sum(sum_of_col)
        #print(self.total_y2d)
        #print(self.total_y1d)
    
    def prospect(self, x, ind, alpha, num_of_feat):
        lst = []
        i=0
        while i <x.shape[0]:
            y2d =self.total_y2d[ind,i]            #finding the likelihood based on alpha value and likelihood formula
            y1d  =self.total_y1d[ind]
            formu=y2d+alpha
            fo1=alpha*num_of_feat
            val=(formu/(y1d + fo1))**x[i]
            lst.append(val)
            i+=1
        #print(lst)
        return np.prod(lst)
    
    def predict(self, x, alpha, num_of_cla, num_of_feat, class_P):        #predicting the classification values for the testing data set
        no_test_samples, no_t_feat = x.shape
        pred_prob = np.zeros((no_test_samples, num_of_cla))
        for i in range(no_test_samples):
            prospect_j = np.zeros((num_of_cla))
            for j in range(num_of_cla):
                prospect_j[j]  = class_P[j] * self.prospect(x[i], j, alpha, num_of_feat) 
            for j in range(num_of_cla):
                pred_prob[i,j] = (prospect_j[j]/np.sum(prospect_j))
        answer_index = np.argmax(pred_prob, axis=1)                                     #returning the values of max indices of the probabaility for each testing dataset
        return answer_index


# Building the Multivariate Naive Bayes model and defining training, testing and likelihood calculation function by inputing class probability, alpha number of features e.t.c


ml_model = Naive_Bayes()
ml_model.model_train(x_train, y_train)
alpha = 8
ypred = ml_model.predict(x_test, alpha, 4, feature_number, class_probability)
print(ypred)
print('Accuracy is:', accuracy_score(ypred, y_test))


# Training the NB model usinf training data set and predicting the result for the testing dataset and finding the accuracy of the result by comparing it with testing dataset

# References
# 
# 
# https://statisticsglobe.com/combine-pandas-dataframes-vertically-horizontally-python
# 
# https://stackoverflow.com/questions/41181779/merging-2-dataframes-vertically
# 
# https://www.geeksforgeeks.org/using-countvectorizer-to-extracting-features-from-text/
# 
# https://machinelearningmastery.com/classification-as-conditional-probability-and-the-naive-bayes-algorithm/
# 
# https://leochoi146.medium.com/how-and-when-to-use-ordinal-encoder-d8b0ef90c28c
# 
# https://medium.com/@johnm.kovachi/implementing-a-multinomial-naive-bayes-classifier-from-scratch-with-python-e70de6a3b92e


