
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class PCA(): 
    def __img_preprocess__(self,img):
        self.mean = np.mean(img)
        self.covar_mat = img-self.mean
        self.ei_val,self.ei_vec = np.linalg.eig(np.cov(self.covar_mat))
        # after calculating eigen vectors and eigen values, we will sort these vectors with respect to their eigen values 
        # this step will reduce sorting our eigen values each time we need to do a component analysis
        indx = np.argsort(self.ei_val)[::-1] # as we need this in decreasing order
        self.ei_vec = self.ei_vec[:,indx]
        self.ei_val = self.ei_val[indx]
    def __init__(self,img):
        self.__img_preprocess__(img)
    def reduce(self,components):
        variance = sum(self.ei_val[:components])/sum(self.ei_val)
        req_vec = self.ei_vec[:,:components] # selecting the required eigen vectors i.e., principle components
        score = np.dot(req_vec,np.dot(req_vec.T,self.covar_mat))+self.mean.T
        return np.uint8(np.absolute(score)),variance


# performing PCA
if __name__ == '__main__':
    img_path = r"C:\Users\Avinash\Desktop\Assingments\AML\hw3_1.jpeg" # change the image path here
    img = np.asarray(Image.open(img_path))
    red,green,blue = img[:,:,0],img[:,:,1],img[:,:,2]
    PCA_red,PCA_green,PCA_blue = PCA(red),PCA(green),PCA(blue)
    components = [1,10,50,100,250,400,500,700]
    var = {'r':[],'g':[],'b':[]}
    for c in components:
        red_t,red_var = PCA_red.reduce(c)
        green_t,green_var = PCA_green.reduce(c)
        blue_t,blue_var = PCA_blue.reduce(c)
        var['r'].append(red_var)
        var['g'].append(green_var)
        var['b'].append(blue_var)
        reconstructed = np.dstack((red_t,green_t,blue_t))
        print("Image reconstructed with {} features:".format(c))
        plt.imshow(Image.fromarray(reconstructed))
        plt.show()

    plt.xlabel("Number of components")
    plt.ylabel("Variance")
    plt.plot(components,var['r'],color='red')
    plt.plot(components,var['g'],color='green')
    plt.plot(components,var['b'],color='blue')
    plt.show()




