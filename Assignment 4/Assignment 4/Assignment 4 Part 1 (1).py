# %%
#importing required libraries
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# %%
df = pd.read_csv("/content/data1.csv")#get_df()
df

# import numpy as np
# Age = [24,53,23,25,32,52,22,43,52,48]
# Salary = [40000,52000,25000,77000,48000,110000,38000,44000,27000,65000]
# College=['Yes','No','No','Yes','Yes','Yes','Yes','No','No','Yes']
# X = np.array([Age, Salary,College])
# dt = {'names':['Age', 'Salary','College'], 'formats':[int, int, str]}
# Y = np.zeros(len(Age), dtype=dt)
# print(dt)
# Y['Age'] = X[0]
# Y['Salary'] = X[1]
# Y['College']=X[2]
# print(Y)

# %%
#function to calculate entropy that belong to a particular class
def entropy_of_classes(one): 
    classEntropy = set(one)  # assigning the classEntropy
    summ = 0
    n = len(one)
    for no_of_class in classEntropy:   # for each class, get entropy
        class_n = sum(one==no_of_class)   #number of classes where elemensts in classEntropy is equal to one
        class_not_n=sum(one!=no_of_class)  #number of classes where elemensts in classEntropy is not equal to one
        #entr_c=entropy_cal(class_n, class_not_n) #calling the entropy functions to calculate the entropy
        if class_n== 0 or class_not_n == 0:  # if either class one is 0 or class two is 0 the entropy is 0
            entr_c= 0
        else:
            combined=class_n+class_not_n               #total number 
            entr_c = -(class_n*1.0/combined)*math.log(class_n*1.0/combined, 2) + -(class_not_n*1.0/combined)*math.log(class_not_n*1.0/combined, 2)
        e = class_n*1.0/n *entr_c  # weighted average
        summ += e       #adding the result to the total ns
        #print(summ,n)
    return summ, n

# %%
# Function to claculate the entropy of both the children in a node
def entropyTotal(y_predict, y_actual):
    len_y=len(y_predict)
    len_act=len(y_actual)
    if len_y != len_act:
        print('Same Length of Y actual and Y predict')
        return None
    n = len(y_actual)
    sideTrue, no_of_true = entropy_of_classes(y_actual[y_predict]) # entropy of left child by calling entropy function
    #print(sideTrue,no_of_true)
    sideFalse, no_of_false = entropy_of_classes(y_actual[~y_predict]) # entropy of right child by calling entropy function
    #print(sideFalse,no_of_false)
    sides = no_of_true*1.0/n * sideTrue + no_of_false*1.0/n * sideFalse # Total entropy with weighted averages
    return sides

# %% [markdown]
# **1.1**

# %%
#class defining Decision Tree Classifier
class DtClassifier(object):
    def __init__(self,namesOfFeatures,maxDepth):
        self.depth = 0
        self.namesOfFeatures=namesOfFeatures # definign the class members
        self.maxDepth = maxDepth
    
    def fit_tree(self, x, y, partitionNode={}, depth=0):
        if partitionNode is None: # there is no partition of node  
            return None # then we return none
        elif len(y) == 0: # or if the length of target varible is 0 we return None
            return None
        elif self.same_items(y): # if the function has the same items
            return {'value_':y[0]} # we return its value_ues
        elif depth >= self.maxDepth: # if the depth of our tree is greater than the maximum depth assigned we return none
            return None
        else: 
            column, threshold, entropy,informationGain = self.best_split(x, y) #based on the information gain we split
            left_split = y[x[:, column] < threshold] # assigning the thresholds to both left and right value_ues of y
            right_split = y[x[:, column] >= threshold]   #assigning  left and right split the values of y based on the threshold
            #print(left_split,right_split)
            partitionNode = {'column': self.namesOfFeatures[column], 'index_col':column, 'threshold':threshold,'value_': np.round(np.mean(y)),
                       'entropy':entropy,'informationGain':informationGain}
            # based on the thresholds assigining the left partitions
            partitionNode['left'] = self.fit_tree(x[x[:, column] < threshold], left_split, {}, depth+1) 
            # based on the thresholds assigining the right partitions
            partitionNode['right'] = self.fit_tree(x[x[:, column] >= threshold], right_split, {}, depth+1)
            self.depth += 1 # adding depth by 1 more level 
            self.trees = partitionNode
            return partitionNode
    
    def best_split(self, x, y):
        informationGainMinimum=0   
        entropyMinimum = 1 # Minimum entropy that we assign is 1
        column = None
        threshold = None  # assigning thereshold as None
        for i, c in enumerate(x.T): #iterating through the avleus in x
            entropy,informationGain,currentThreshold=self.helper_split(c, y)
            if entropy == 0:    # We stop iterating after we have found the very first threshold
                return i, currentThreshold, entropy,informationGain
            if entropy <= entropyMinimum: # if the entropy is lesser or equal than the minimum 
                entropyMinimum = entropy # we assign teh current entorpy as teh minimum entropy
                informationGainMinimum=informationGain ## we assign the informationGain as informationGainMinimum
                column = i
                threshold = currentThreshold # we assign the currentThreshold as threshold
        return column, threshold, entropyMinimum,informationGainMinimum
    
#function for finding the best split based on the target and input parameters
    def helper_split(self, column, y):
        entropyMinimum = 15 # assigning minimum entropy as 10
        entropyOfParent=entropy_of_classes(y)[0] #assigning the first class entropy as the parent entropy
        n = len(y) #length of y
        for value_ue in set(column):
            y_predict = column < value_ue # assiging teh boolean value to the y_predict if column value is lesser than the value 
            entropyNow = entropyTotal(y_predict, y) # assigning the previous entropy and our currrent entropy
            if entropyNow <= entropyMinimum:
                informationGainMinimum = entropyOfParent-entropyNow # calculating informationGainMinimum
                entropyMinimum = entropyNow
                threshold = value_ue # assign teh current avlue as threshold
        return entropyMinimum, informationGainMinimum, threshold
    
    def same_items(self, items): # returning all the values in items
        return all(x == items[0] for x in items)
    
#function for doing thh predictions                                       
    def predict(self, x):
        res_ = np.array([0]*len(x)) # assigning an array of zeros to the results list
        tree = self.trees
        for i, c in enumerate(x):
            res_[i] = self.predict_helper(c)#  calculating the values of results
        return res_

#function to get the prediction
    def predict_helper(self, row):
        currentLevel = self.trees
        while currentLevel.get('threshold'):# if the current levl of threshold is greater than therow of teh current level in teh indec xolumns
            if row[currentLevel['index_col']] > currentLevel['threshold']:
              #print('right')
                currentLevel = currentLevel['right'] # we assign the right child as the current level or                
            else:
                #print('left'')
                currentLevel = currentLevel['left'] # we assign the left child as the current level

        else:
            return currentLevel.get('value_')


# %%
def last_col_conv(amount):
    if amount == 'Yes':
        #if the value is Yes returning one else zero
        return 1
    return 0
df.iloc[:,-1]=df.iloc[:,-1].map(last_col_conv)   #mapping the last column values to the 1 or 0 based on Yes or No
first=np.array(df.iloc[:,0:2])

# %%
tree = DtClassifier(df.columns,maxDepth=5)
inpu=np.array(df.iloc[:,0:2])    #taking the x values in the data 
res=np.array(df.iloc[:,-1])       #y values in the dataframe
m = tree.fit_tree(inpu, res)      #fitting the value into the Tree 
print(m) 

# %%
data_frame_1=df.copy() # copying the data frame 
data_frame_1['predict']=-100
ent_of_par=entropy_of_classes(res)[0] # parent's entropy 
value_of_gain=0   #initializing the min value of gain ot 0
np.random.seed(1)
alpha = np.random.rand() # assigning some random variables to alpha and beta
beta=np.random.rand()   
countParent=res.shape[0]   #shaping the parent variable into the shape of y

# %%
for loop in range(100):
    for i in data_frame_1.index:
        x1=data_frame_1.iloc[i].values[0] # assiging the 0th values from the data_frame_1 dataframe to x1
        x2=data_frame_1.iloc[i].values[1] # assiging the 1th values from the data_frame_1 dataframe to x2
        if alpha*x1+beta*x2-1<0: # basic check
            data_frame_1.loc[i,'predict']=0 # if the values of the index innadat frame an dtaht of the predict are 0 then
            if(data_frame_1.loc[i,'predict']!=data_frame_1.loc[i,'College Degree']): # we check if they are not equal to the colleg degree values
                alpha=alpha+x1 # if not then we add the x values from the data frame to alpha values
                beta=beta+x2 # same it is for beta
        else:
            data_frame_1.loc[i,'predict']=1 # if teh values of the index in data frame and tht of the predict are 1 then
            if(data_frame_1.loc[i,'predict']!=data_frame_1.loc[i,'College Degree']): # if prediction is not equal to the true value the going into the loop
                alpha=alpha-x1 #  subtract the x1 values from the data frame from aplha vaue
                beta=beta-x2 # subtracting x2 from beta
    
 #assigning the left and right values of the tree based on the target variable   
    left_side=data_frame_1.loc[data_frame_1['predict']==0,'College Degree']    #if the college degree is 0, going left
    right_side=data_frame_1.loc[data_frame_1['predict']==1,'College Degree']    #if the college degree is 1, going right
    entropyRight=entropy_of_classes(right_side)[0] #assigning the left and right values of the entropy 
    rightCount=right_side.shape[0]     #finding the shape of right dataframe (number of nodes in right side)
    leftCount=left_side.shape[0] # assigning the rows count as the left child dataframe
    #print(rightCount,leftCount)
    entropyLeft=entropy_of_classes(left_side)[0]   #finding the entropy of the left chidren
    left=entropyLeft*(leftCount/countParent)   #left side value inorder to find the gain
    right=entropyRight*(rightCount/countParent)  #left side value inorder to find the gain
    #print(left,right)
    averages=left+right      
    value_of_gain=ent_of_par-averages # calculating the gain by subtracting average from entropy of parent
    if(value_of_gain==ent_of_par):   #exiting the loop when gain is equal to entropy of class
      print(" Final Alpha Values=",alpha, "; Beta Values=",beta, "; value_of_gain Values=",value_of_gain)
      break


# %%
fig = plt.figure()
#setting up size of the figure
fig.set_size_inches(15,10)
ax = plt.axes()
#giving the figure the data, x, y and the result values
ax=sns.scatterplot(data=df,x='Age',y='Salary',hue='College Degree')   #doing the scatter plot
xx=np.linspace(20,55,2)
ax.plot(xx,(-1*(1/beta)-(alpha/beta)*xx))    #plotting the classification line based on the values alpha and beta
plt.show()

# %% [markdown]
# **1.3**

# %% [markdown]
# **Advantages** 
# 
# A single variate or variable quantity is what is meant by the term "univariate." Contrarily, the term "multivariate" refers to the simultaneous use of several variates or variable quantities. Multivariate decision trees can therefore branch using all qualities at a single node. Multivariate decision trees converge substantially more quickly for the linearly separable datasets than univarate decision trees.
# 
# **Disadvantages**
# 
# In multivariate trees, hyperplanes of any orientation are employed. This implies that there could be 2^d(NCd)  different Hyperplanes. This makes thorough search ineffective and unworkable. Therefore, utilizing a linear multivariate node that adds the weights for each feature is a more useful method of using. In order to make the procedure more effective and practical, linear multivariate decision trees select the most crucial features out of all of the available ones.Additionally, the fundamental use of univariate decision trees is to make the feature importance more interpretable than it would be with multivariate trees.

# %% [markdown]
# 1.3
# 
# Advantages:
# 
# Univariate - having one variate or variable quantity. 
# Multivariate - having multiple or more than one variate or variable quantity at the same time. Multivariate decision trees  consume all the features of one node when branching. Also, for the datasets that are linearly separable, multivariate decision trees converge quicker than univarate decision trees.
# 
# Disadvantages:
# 
# The Hyperplanes that have arbitrary orientation are usedin multivariate trees which says there can be $2^d$ ${N \choose d}$ possible Hyperplanes.
# 
# (d is the number of dimensions and N is the number of possible thresholds for the split points)
# 
# making exhaustive search inefficient and impractical,a practical way is using linear multivariate node which takes weights for every attribute and sums them up .Linear multivariate decision trees hold the very important features out of all making the process more efficient and practical.
# 
# Univariate decision trees are used for interpreting the feature importances which isnt applicable for multivariate trees.

# %%


# %% [markdown]
# *References:*
# <li>https://towardsdatascience.com/implementing-a-decision-tree-from-scratch-f5358ff9c4bb
# <li>https://web.njit.edu/~usman/courses/cs675_fall16/Comparing_Univariate_and_Multivariate_De.pdf
# <li>https://anderfernandez.com/en/blog/code-decision-tree-python-from-scratch/
# <li>https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/ (for splitting the tree)
# <li>https://levelup.gitconnected.com/building-a-decision-tree-from-scratch-in-python-machine-learning-from-scratch-part-ii-6e2e56265b19
# </li>

# %% [markdown]
# 

# %%


# %%



