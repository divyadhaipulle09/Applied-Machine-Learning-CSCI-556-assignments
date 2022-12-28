# %%
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier


# %% [markdown]
# <h5>Bagging</h5>

# %%
class Bagging(object):
    def __init__(self,max_depth,seed=None):
        self.classifiers = []
        self.seed = seed
        self.tree_depth = max_depth
        self.n = None
        self.d = None
        np.random.seed(self.seed)

    def fit(self,X_train,y_train,bootstrap):
        
        self.bootstrap = bootstrap
        self.n , self.d = X_train.shape
        for b in range(self.bootstrap):
            sample = np.random.choice(np.arange(self.n),size = self.n,replace=True)
            temp_X_train = X_train[sample]
            temp_y_train = y_train[sample]
            weak_class = DecisionTreeClassifier(max_depth=self.tree_depth)
            weak_class.fit(temp_X_train, temp_y_train)
            self.classifiers.append(weak_class)
    def predict(self,X_test):
        len_xt = len(X_test)
        y_hat = np.empty((len(self.classifiers),len_xt))
        y_pred = np.empty(len_xt)
        for i,weak_class in enumerate(self.classifiers):
            y_hat[i] = weak_class.predict(X_test)
        
        return np.where(y_hat.sum(0)>0,1,-1)

# %% [markdown]
# <h5>Adaboost</h5>

# %%
class AdaBoost(object):
    def __init__(self,max_depth):
        self.tree_depth = max_depth
        self.training_err = []
        self.pred_err = []
        self.rounds = None
        self.betas = []
        self.classifiers = []
    def fit(self,X,y_true,rounds = 100):
        self.betas = []
        self.training_err = []
        self.rounds = rounds
        weights = np.ones(len(y_true)) * 1/len(y_true)
        for round in range(rounds):
            if round != 0:
                # Step-4 Update the current weights
                weights = self.change_weights(weights,beta,y_true,y_pred)
            # Step-1 Train and fit a weak classifier
            weak_class = DecisionTreeClassifier(max_depth = self.tree_depth)
            weak_class.fit(X,y_true,sample_weight = weights)
            y_pred = weak_class.predict(X)
            self.classifiers.append(weak_class)

            # Step-2 Calculate the error of our weak classifier
            temp_err = self.calculate_err(weights,y_true,y_pred)
            self.training_err.append(temp_err)

            # Step-3 Calculate the new beta and append it to our beta table
            beta = self.calculate_beta(temp_err)
            self.betas.append(beta)
        assert len(self.betas) == len(self.classifiers)
    def calculate_err(self,weights,y_true,y_pred):
        return (sum(weights*(np.not_equal(y_true,y_pred)).astype(int)))/sum(weights)
    def calculate_beta(self,err):
        return np.log((1-err)/err)
    def change_weights(self,weights,beta,y_true,y_pred):
        
        return weights*np.exp(beta*(np.not_equal(y_true,y_pred)).astype(int))
    def predict(self,X):
        weak_pred = pd.DataFrame(index = range(len(X)),columns=range(self.rounds))

        for round in range(self.rounds):
            temp_pred = self.classifiers[round].predict(X) * self.betas[round]
            weak_pred.iloc[:,round] = temp_pred
        actual_pred = (1*np.sign(weak_pred.T.sum())).astype(int)
        return actual_pred

# %%
# re-usable ploting function for our ensemble models i.e., bagging, boosting
def plot(ens_model,X_train,y_train,X_test,y_test):
    train_err = np.empty(len(ens_model.classifiers))
    test_err = np.empty(len(ens_model.classifiers))
    for i,weak_class in enumerate(ens_model.classifiers):
        test_err[i] = 1-(len(y_test[weak_class.predict(X_test)==y_test])/len(y_test))
        train_err[i] = 1-(len(y_train[weak_class.predict(X_train)==y_train])/len(y_train))
    plt.figure(figsize=(16,8))
    plt.plot(train_err,c='g')
    plt.plot(test_err,c='r')
    plt.ylabel('Error')
    plt.xlabel('Bootstrap/Rounds')
    plt.legend(['train error','test error'],loc='center right')
    plt.show()

# %%
# importing Letter Dataset
df = pd.read_csv('letter-recognition.data',header = None)
df =  df[(df.iloc[:,0]=='C') | (df.iloc[:,0]=='G')]
df[0] = df[0].astype('category').cat.codes

# %%
X_train,X_test,y_train,y_test = train_test_split(df.iloc[:,1:17].values,df.iloc[:,0].values,random_state=54)
y_train = np.where(y_train==0,-1,1)
y_test = np.where(y_test==0,-1,1)


# %%
# Bagging on letter dataset
# setting our weak classifiers as tree stumps
bag_model = Bagging(max_depth=2,seed=456)
bag_model.fit(X_train,y_train,100)
y_pred = bag_model.predict(X_test)
acc = accuracy_score(y_test,y_pred)
print('Accuracy - {0:.2f}% for stumps with 100 rounds'.format(acc*100))

# %%
plot(bag_model,X_train,y_train,X_test,y_test)

# %%
# Lets perform bagging with tree depth=50
bag_model2 = Bagging(max_depth=50,seed=123)
bag_model2.fit(X_train,y_train,200)
y_pred = bag_model2.predict(X_test)
acc = accuracy_score(y_test,y_pred)
print('Accuracy - {0:.2f}% for stumps with 100 rounds'.format(acc*100))

# %%
# with the increase of tree depth the bagging model
# gained a performance boost of over 10%
plot(bag_model2,X_train,y_train,X_test,y_test)


# %%
# performing AdaBoost with stump trees
adaboost_model = AdaBoost(max_depth =2)
adaboost_model.fit(X_train,y_train,rounds=100)
acc = accuracy_score(y_test,adaboost_model.predict(X_test))
print('Accuracy - {0:.2f}% for stumps with 100 rounds'.format(acc*100))

# %%

plot(adaboost_model,X_train,y_train,X_test,y_test)


# %% [markdown]
# we can see that a hand full of our weak classifiers(stump trees) have accuracy score less than 50%, which is worse that guessing a coin toss. But due to the power of boosting we have achieved ~98% accuracy 

# %%
#  performing AdaBoost with 'good' tree depth
adaboost_model = AdaBoost(max_depth = 8)
adaboost_model.fit(X_train,y_train,rounds=100)
acc = accuracy_score(y_test,adaboost_model.predict(X_test))
print('Accuracy - {0:.2f}% with 100 rounds'.format(acc*100))

# %%
# with increased tree depth we gained 1% accuracy in performance, lets plot our training and testing errors
plot(adaboost_model,X_train,y_train,X_test,y_test)


# %%
df = pd.read_csv('spambase.data',header=None)

df[57] = df[57]*2-1
X_train, X_test, y_train, y_test = train_test_split((df.iloc[:,0:57].values),(df.iloc[:,57].values),random_state=123)

# %%
# Boosting on Spam Dataset
ada_model = AdaBoost(max_depth =2)
ada_model.fit(X_train,y_train,100)
pred = ada_model.predict(X_test)
acc = accuracy_score(y_test,ada_model.predict(X_test))
print('Accuracy - {0:.2f}% with 100 rounds'.format(acc*100))

# %%
# along with accuracy score lets check the confusion matrix of our boosting model
print(confusion_matrix(y_test,pred))
# from confusion matrix we can calculate the precision, recall, and f1 score of our matrix

# %%
# so we will use sklearn inbuilt method to provide us the above mentioned metrocs
print(classification_report(y_test,pred))

# %%
# lets perform adaboost with deeper trees
ada_model2 = AdaBoost(max_depth = 6)
ada_model2.fit(X_train,y_train,100)
pred = ada_model2.predict(X_test)
acc = accuracy_score(y_test,ada_model2.predict(X_test))
print('Accuracy - {0:.2f}% with 100 rounds'.format(acc*100))

# %%
# with the increase of tree depth we have gained ~1% in classification performance
# lets plot the training,testing errors of both models
plot(ada_model,X_train,y_train,X_test,y_test)
plot(ada_model2,X_train,y_train,X_test,y_test)
# we can observe that with increase in tree depth the mean error rate of individual model has decreased a lot

# %%
# Bagging for Spam dataset
# setting our weak classifiers as tree stumps
bag_model = Bagging(max_depth=2,seed=456)
bag_model.fit(X_train,y_train,100)
y_pred = bag_model.predict(X_test)
acc = accuracy_score(y_test,y_pred)
print('Accuracy - {0:.2f}% for stumps with 100 rounds'.format(acc*100))


# %%
# Lets perform bagging with tree depth=50
bag_model2 = Bagging(max_depth=50,seed=123)
bag_model2.fit(X_train,y_train,200)
y_pred = bag_model2.predict(X_test)
acc = accuracy_score(y_test,y_pred)
print('Accuracy - {0:.2f}% for stumps with 100 rounds'.format(acc*100))

# %%
# lets check the confusion matrix and classification report of the better performing bagging model
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
# we can observe that varaince has reduced with the better performing bagging model

# %%
# lets plot the training,testing errors of both models
plot(bag_model,X_train,y_train,X_test,y_test)
plot(bag_model2,X_train,y_train,X_test,y_test)

# %% [markdown]
# even though the 1% testing error falls high for second model, it still does a better job in classification. We should be careful in increasing the tree depth as it may lead to overfitting

# %%

df = pd.read_fwf('german.data-numeric',header=None)
df[24]=df[24].apply(lambda x:1 if x!=1 else -1)
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,0:24].values, 
                                                    df.iloc[:,24].values, 
                                                    random_state = 123)
# np.unique(y_train)                                                    

# %%
# Boosting on German credit data
ada_model = AdaBoost(max_depth =1)
ada_model.fit(X_train,y_train,100)
pred = ada_model.predict(X_test)
acc = accuracy_score(y_test,ada_model.predict(X_test))
print('Accuracy - {0:.2f}% with 100 rounds'.format(acc*100))

# %%
# lets perform adaboost with deeper trees
ada_model2 = AdaBoost(max_depth = 4)
ada_model2.fit(X_train,y_train,100)
pred = ada_model2.predict(X_test)
acc = accuracy_score(y_test,ada_model2.predict(X_test))
print('Accuracy - {0:.2f}% with 100 rounds'.format(acc*100))

# %%
# from the above runs we can observe that the performance gain is minimal compared compared to the computational cost comes with deeper trees,
# this could be due to the fact that our model migth be overfitting
plot(ada_model,X_train,y_train,X_test,y_test)
plot(ada_model2,X_train,y_train,X_test,y_test)

# %%
# Bagging 
# setting our weak classifiers as tree stumps
bag_model = Bagging(max_depth=2,seed=456)
bag_model.fit(X_train,y_train,100)
y_pred = bag_model.predict(X_test)
acc = accuracy_score(y_test,y_pred)
print('Accuracy - {0:.2f}% for stumps with 100 rounds'.format(acc*100))


# %%
bag_model2 = Bagging(max_depth=50,seed=123)
bag_model2.fit(X_train,y_train,200)
y_pred = bag_model2.predict(X_test)
acc = accuracy_score(y_test,y_pred)
print('Accuracy - {0:.2f}% for stumps with 200 rounds'.format(acc*100))

# %%
# lets plot the training,testing errors of both models
plot(bag_model,X_train,y_train,X_test,y_test)
plot(bag_model2,X_train,y_train,X_test,y_test)

# %% [markdown]
# In bagging for german credit dataset, we can observe that for stump trees both our train, test error are nearly identical but for deeper trees there is a huge difference between test train errors this could be due to our model started to overfit, lets try bagging with little less depth

# %%
bag_model3 = Bagging(max_depth=50,seed=123)
bag_model3.fit(X_train,y_train,50)
y_pred = bag_model3.predict(X_test)
acc = accuracy_score(y_test,y_pred)
print('Accuracy - {0:.2f}% for stumps with 100 rounds'.format(acc*100))
plot(bag_model3,X_train,y_train,X_test,y_test)

# %% [markdown]
# we can notice that our testing errors are not as huge as before with 100 bootstrap and 50 tree depth

# %% [markdown]
# <h4>References</h4>
# <ol>
# <li><div>Datsets used</div>
#     <ul>
#         <li>Spam base-https://archive.ics.uci.edu/ml/datasets/spambase</li>
#         <li>German Credit-https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)</li>
#         <li>Letter recognition-https://archive.ics.uci.edu/ml/datasets/letter+recognition</li>
#     </ul>
# </li>
# <li>https://www.section.io/engineering-education/implementing-bagging-algorithms-in-python/</li>
# <li>https://dafriedman97.github.io/mlbook/content/c6/s2/bagging.html</li>
# <li>https://towardsdatascience.com/adaboost-from-scratch-37a936da3d50</li>
# </ol>


