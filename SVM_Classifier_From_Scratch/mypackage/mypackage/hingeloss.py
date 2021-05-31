import numpy as np

def loss_old(feature_matrix, labels, W, b,verbose=False):
    """
    #######################################################
    #
    #   Function to compute total hinge loss
    #
    #######################################################
    """    
	
    def loss_single(feature_vector, label, W, b,verbose):
	    """
		#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
		#
		# Helper function to compute single data point hinge loss
		#
		#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
		"""
	    ydash = (np.dot(W,feature_vector) + b)
        
	    hinge = np.max([0.0, 1 - ydash*label])
        
	    if verbose:
	      print('Label: %d | W.X + b: %f | 1 - y*(W.X + b): %f | hinge: %f'%(label,ydash,1 - ydash*label,hinge))
	    return hinge

    loss_total = 0
    num = len(feature_matrix)
    for i in range(num):
        loss_total += loss_single(feature_matrix[i], labels[i], W, b,verbose)
    hinge_new=list(map(lambda v : max(0,v),1-(np.matmul(W,feature_matrix.T)- b)*labels))
    print(sum(hinge_new))
    return loss_total

def loss(feature_matrix, labels, W, b,verbose=False):
    """
    #######################################################
    #
    #   Function to compute total hinge loss
    #
    #######################################################
    """    
    loss_total=list(map(lambda v : max(0,v),1-(np.matmul(W,feature_matrix.T)- b)*labels))
    loss_total=sum(loss_total)
    return loss_total

def error_x(X, y, W, b,verbose=False):
    """
    #######################################################
    #
    #   Function to display misclassified data points
	#	( could be used to compute total hinge low faster )
    #
    #######################################################
    """
    hinge=list(map(lambda v : max(0,v),1-(np.matmul(W,X.T)- b)*y))
    if sum(hinge)>0.001:
        print('--------- Misclassified data points. ----------')
        for x, h,l in zip(X,hinge,y):
            if h >0:
                print('x: %s | y: %d | hinge: %f'%(str(x),l,h))
    else:print('--------- No misclassified data point. ---------')
