import os.path
from os import path


import torch
import torch.nn as nn
import torch.optim as optim

import random

from mypackage import DataLoader as DL, NN, Attacks

torch.manual_seed(0)
random.seed(0)

# ########## Load datasets with batch_size=100 ##########
train_loader, test_loader = DL.Loader(100).load_train_test()

# ########## Set path of the saved models ##########
saved_models_path = 'saved_models/'

# ########## Define the target device (CPU or GPU) to manage tensors ##########
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('\nInfo : Device used %s'%device)

# ########## Define the loss function ##########
criterion = nn.CrossEntropyLoss() 


#print("\n 1.1 - //////////////// Training Linear Fully Connected NN  and Conv2D NN ////////////////\n")
print("""
# ######################################################################################
#
#  Step 1.1
#           - If trained versions of a Fully Connected Linear NN and a Conv2D NN exist
#               => instanciate and load them
#
#           - Otherwise .
#                => instanciate, train and save them
#
# ######################################################################################
""")

if path.isfile(saved_models_path+'fc_model.pt') and path.isfile(saved_models_path+'conv_model.pt') :
	"""
	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	#
	#						If exist, Load FC Linear Model
	#
	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	"""	 
	fc_model=NN.FullyConnectedModel(input_dim = 784,
                          output_dim = 10,
                          hidden_dim =100)
	if torch.cuda.is_available() == False:
		fc_model.load_state_dict(torch.load(saved_models_path+"fc_model.pt", map_location=torch.device('cpu')))
	else : fc_model.load_state_dict(torch.load(saved_models_path+"fc_model.pt"))
	fc_model.to(device)
	fc_model.eval()
	print('\n --- FC_Model already trained ---\n\n\t%s'%str(fc_model))
	
	"""
	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	#
	#						If exist, Load Conv Model
	#
	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	"""	     
	conv_model=NN.ConvModel(input_dim = 28,
                          output_dim = 10,
                          hidden_dim =64,
                          num_filters = 32,
                          kernel_size = 5,
                          stride = 1,
                          padding=2,
                          pool_size=2)
	if torch.cuda.is_available() == False:
		conv_model.load_state_dict(torch.load(saved_models_path+"conv_model.pt", map_location=torch.device('cpu')))
	else:conv_model.load_state_dict(torch.load(saved_models_path+"conv_model.pt"))
	conv_model.to(device)
	conv_model.eval()
	print('\n --- Conv_Model already trained ---\n\n\t%s'%str(conv_model))
    
else:
	print("\n ---  Training Fully Connected Linear Model ---\n")
	"""
	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	#
	#						If not exist, train and save FC Linear Model
	#
	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	"""	      
	fc_model = NN.FullyConnectedModel(input_dim = 784,
                              output_dim = 10,
                              hidden_dim =100).to(device)
    
	opt = optim.SGD(fc_model.parameters(), lr=.01, momentum=0.9, weight_decay=1e-04) # define your optimizer
	criterion = nn.CrossEntropyLoss() # define your loss function
	NN.train_model(fc_model.to(device), criterion, opt, train_loader, epochs = 10)	    
	model_name = saved_models_path+"fc_model.pt"
	torch.save(fc_model.state_dict(), model_name)
	
	print("\n --- Training Conv2D Model ---\n")
	"""
	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	#
	#						If not exist, train and save Conv2D Model
	#
	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	"""	 
	conv_model = NN.ConvModel(input_dim = 28,
                              output_dim = 10,
                              hidden_dim =64,
                              num_filters = 32,
                              kernel_size = 5,
                              stride = 1,
                              padding=2,
                              pool_size=2)
	opt = optim.SGD(conv_model.parameters(), lr=.01, momentum=0.9, weight_decay=1e-04)
	criterion = nn.CrossEntropyLoss() # define your loss function
	NN.train_model(conv_model.to(device), criterion, opt, train_loader, epochs = 10)	    
	model_name = saved_models_path+"conv_model.pt"
	torch.save(conv_model.state_dict(), model_name)





#print("\n 1.2 - //////////////// Linear Fully Connected NN  and Conv2D NN  Evaluation - Before and After FGSM/PGD Attacks ////////////////\n")
print("""
# #######################################################################################
#
#  Step 1.2
#           - Evaluate each NN on the Test Set
#              a - Standard evaluation
#              b - Evaluation when attacked by FGSM algo
#              c - Evaluation when attacked by PGD algo
#              
# #######################################################################################
""")

"""
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#
#					Evaluation of FC Linear Model
#
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
"""
"""
# ----------------------------------------------------------------------
#					Define Epsilone, FGSM Attack and PGD Attack
# ----------------------------------------------------------------------
"""
# ########## Define Epsilon (intensity of Perturbation) ##########
epsilon = 0.1 

# ########## Instanciate FGSM Attack ##########
fgsm = Attacks.FastGradientSignMethod(fc_model, epsilon, criterion)

# ########## Instanciate PGD Attack ##########
pgd = Attacks.ProjectedGradientDescent(fc_model, eps=.1,
										alpha=.1, num_iter=10,
										random=True, criterion=criterion)
										
"""
# ----------------------------------------------------------------------
#			Evaluation : a - without Attack, b - under FGSM Attack, c - under PGD Attack
# ----------------------------------------------------------------------
"""																			

# a - ########## Loss of the FC Linear Model (No attack) ##########
loss_wo_attack, acc_wo_attack = NN.eval_model(fc_model, test_loader,criterion)

# b - ########## Loss of the FC Linear Model under FGSM attack ##########
loss_fgsm_attack, acc_fgsm_attack, fc_fgsm_delta = NN.eval_model(fc_model, test_loader,criterion, fgsm)

# c - ########## Loss of the FC Linear Model under PGD attack ##########
loss_pgd_attack, accuracy_pgd_attack, fc_pgd_delta = NN.eval_model(fc_model, test_loader,criterion, pgd)
#///////////////////////////////////////////////////////////////////////

print("\n --- FC model ---\n")
print(f"\t Test Evaluation - Before Attack | Loss: {loss_wo_attack:.4f} \t | Accuracy: {acc_wo_attack:.2%}")
print(f"\t Test Evaluation - After FGSM Attack |  Loss: {loss_fgsm_attack:.4f} \t | Accuracy: {acc_fgsm_attack:.2%} | Accuracy Gap FGSM: {(acc_fgsm_attack - acc_wo_attack):.2%}")
print(f"\t Test Evaluation - After PGD Attack |  Loss: {loss_pgd_attack:.4f} \t | Accuracy: {accuracy_pgd_attack:.2%} | Accuracy Gap PGD: {(accuracy_pgd_attack - acc_wo_attack):.2%}")

"""
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#
#					Evaluation of FC Linear Model
#
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
"""
"""
# ----------------------------------------------------------------------
#					Define FGSM Attack and PGD Attack
# ----------------------------------------------------------------------
"""
# ########## Instanciate FGSM Attack ##########
fgsm = Attacks.FastGradientSignMethod(conv_model, epsilon, criterion)

# ########## Instanciate PGD Attack ##########
pgd = Attacks.ProjectedGradientDescent(conv_model, eps=.1, alpha=.1, num_iter=10, random=True, criterion=criterion)

"""
# ----------------------------------------------------------------------
#			Evaluation : a - without Attack, b - under FGSM Attack, c - under PGD Attack
# ----------------------------------------------------------------------
"""
# a - #############		
loss_wo_attack, acc_wo_attack = NN.eval_model(conv_model, test_loader,criterion)
# b - #############
loss_fgsm_attack, acc_fgsm_attack, cnn_fgsm_delta = NN.eval_model(conv_model, test_loader,criterion, fgsm)
# c - ############# 
loss_pgd_attack, accuracy_pgd_attack, cnn_pgd_delta = NN.eval_model(conv_model, test_loader, criterion,pgd)

print("\n --- CNN model ---\n")
print(f"\t Test Evaluation - Before FGSM Attack | Loss: {loss_wo_attack:4f} \t | Accuracy: {acc_wo_attack:2%}" )
print(f"\t Test Evaluation - After FGSM Attack |  Loss: {loss_fgsm_attack:4f} \t | Accuracy: {acc_fgsm_attack:2%} | Accuracy Gap FGSM: {(acc_fgsm_attack - acc_wo_attack):.2%}")
print(f"\t Test Evaluation - After PGD Attack |  Loss: {loss_pgd_attack:4f} \t | Accuracy: {accuracy_pgd_attack:2%} | Accuracy Gap PGD: {(accuracy_pgd_attack - acc_wo_attack):.2%}")


#print("\n 2.1 - ////////////////  Training Linear Fully Connected NN and Conv2D NN - Under FGSM Attack for more resilience ------------------\n")
print(
"""
# ######################################################################################
#
#  Step 2.1
#           - If trained against FGSM versions of the NN exist
#               => instanciate and load them
#
#           - Otherwise .
#                => instanciate, train under FGSM attack and save them
# 
# ######################################################################################
""")

if path.exists(saved_models_path+'fc_model_vs_FGM.pt') and path.exists(saved_models_path+'fc_model_vs_FGM.pt') :
	"""
	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	#
	#						If exist, load robust FC Linear Model
	#
	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	"""	

	fc_model_vs_FGM=NN.FullyConnectedModel(input_dim = 784,
                          output_dim = 10,
                          hidden_dim =100)
	if torch.cuda.is_available() == False:
		fc_model_vs_FGM.load_state_dict(torch.load(saved_models_path+'fc_model_vs_FGM.pt', map_location=torch.device('cpu')))
	else :  fc_model_vs_FGM.load_state_dict(torch.load(saved_models_path+'fc_model_vs_FGM.pt'))
	fc_model_vs_FGM.to(device)
	fc_model_vs_FGM.eval()
	print('\n --- Info  FC_model_vs_FGM already trained \n\n\t%s'%fc_model_vs_FGM)
    
	"""
	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	#
	#						If exist, load robustConv Model
	#
	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	"""	 

	conv_model_vs_FGM=NN.ConvModel(input_dim = 28,
                          output_dim = 10,
                          hidden_dim =64,
                          num_filters = 32,
                          kernel_size = 5,
                          stride = 1,
                          padding=2,
                          pool_size=2)
	if torch.cuda.is_available() == False:
		conv_model_vs_FGM.load_state_dict(torch.load(saved_models_path+'conv_model_vs_FGM.pt', map_location=torch.device('cpu')))
	else :    conv_model_vs_FGM.load_state_dict(torch.load(saved_models_path+'conv_model_vs_FGM.pt'))
	conv_model_vs_FGM.to(device)
	conv_model_vs_FGM.eval()
	print('\n --- Info  Conv_model_vs_FGM already trained -------\n\t\t%s'%str(conv_model_vs_FGM))
else:
	print("\n--------------- Training Fully Connected Linear Model - Under FGSM Attack for more resilience -----------------------\n")
	"""
	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	#
	#						If not exist, train FC Linear Model against FGSM and save
	#
	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	"""	          
	fc_model_vs_FGM = NN.FullyConnectedModel(input_dim = 784,
                              output_dim = 10,
                              hidden_dim =100).to(device)
    
	opt = optim.SGD(fc_model_vs_FGM.parameters(), lr=.01, momentum=0.9, weight_decay=1e-04) # define your optimizer
	criterion = nn.CrossEntropyLoss() # define your loss function
	NN.train_model(fc_model_vs_FGM.to(device), criterion, opt, train_loader, epochs = 10,attack=fgsm)
	model_name = saved_models_path+"fc_model_vs_FGM.pt"
	torch.save(fc_model_vs_FGM.state_dict(), model_name)
    
	print("\n--------------- Training Conv2D - Under FGSM Attack for more resilience -----------------------\n")
	"""
	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	#
	#						If not exist, train Conv2D against FGSM and save
	#
	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	"""	 
	conv_model_vs_FGM = NN.ConvModel(input_dim = 28,
                              output_dim = 10,
                              hidden_dim =64,
                              num_filters = 32,
                              kernel_size = 5,
                              stride = 1,
                              padding=2,
                              pool_size=2)
	opt = optim.SGD(conv_model_vs_FGM.parameters(), lr=.01, momentum=0.9, weight_decay=1e-04)
	criterion = nn.CrossEntropyLoss() # define your loss function
	NN.train_model(conv_model_vs_FGM.to(device), criterion, opt, train_loader, epochs = 10,attack=fgsm)
	model_name = saved_models_path+"conv_model_vs_FGM.pt"
	torch.save(conv_model_vs_FGM.state_dict(), model_name)

	
#print("\n 2.2 - //////////////// Resilient Models Evaluation - Before and After FGSM Attacks ////////////////\n")
print(
"""    
# #######################################################################################
#
#  Step 2.2
#           - Evaluate on the Test Set each NN trained against FGSM attack 
#              a - Standard evaluation (no attack)
#              b - Evaluation when attacked by FGSM algo
#              c - Evaluation when attacked by PGD algo 
#                  to check if a FGSM trained model is robuste against other attacks
#              
# #######################################################################################
""")

"""
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#
#					Evaluation of FC Linear Model
#
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
"""
"""
# ----------------------------------------------------------------------
#
#					Define FGSM Attack and PGD Attack
#						PGD will be used to see if a model trained 
#						against FGSM is ressilient against PGD
#
# ----------------------------------------------------------------------
"""

# ########## Instanciate FGSM Attack ##########
fgsm = Attacks.FastGradientSignMethod(fc_model, epsilon, criterion)

# ########## Instanciate PGD Attack ##########
pgd = Attacks.ProjectedGradientDescent(fc_model, eps=.1, alpha=.1, num_iter=10, random=True, criterion=criterion)
"""
# ----------------------------------------------------------------------
#			Evaluation : a - without Attack, b - under FGSM Attack, c - under PGD Attack
# ----------------------------------------------------------------------
"""
# a - #############	
loss_wo_attack, acc_wo_attack = NN.eval_model(fc_model_vs_FGM, test_loader,criterion)
# b - #############	
loss_fgsm_attack, acc_fgsm_attack, fc_fgsm_delta = NN.eval_model(fc_model_vs_FGM, test_loader,criterion, fgsm)
# c - #############	
loss_pgd_attack, accuracy_pgd_attack, fc_pgd_delta = NN.eval_model(fc_model_vs_FGM, test_loader,criterion, pgd)

print("\n --- FC FGSM Resilient Model ---\n")
print(f"\t Test Evaluation - Model Trained against FGSM Attack | Loss: {loss_wo_attack:.4f} \t | Accuracy: {acc_wo_attack:.2%}")
print(f"\t Test Evaluation - After FGSM Attack |  Loss: {loss_fgsm_attack:.4f} \t | Accuracy: {acc_fgsm_attack:.2%} | Accuracy Gap FGSM: {(acc_fgsm_attack - acc_wo_attack):.2%}")
print(f"\t Test Evaluation - After PGD Attack |  Loss: {loss_pgd_attack:.4f} \t | Accuracy: {accuracy_pgd_attack:.2%} | Accuracy Gap PGD: {(accuracy_pgd_attack - acc_wo_attack):.2%}")

"""
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#
#					Evaluation of CONV2D
#
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
"""
"""
# ----------------------------------------------------------------------
#
#					Define FGSM Attack and PGD Attack
#						PGD will be used to see if a model trained 
#						against FGSM is ressilient against PGD
#
# ----------------------------------------------------------------------
"""
# ########## Instanciate FGSM Attack ##########
fgsm = Attacks.FastGradientSignMethod(conv_model, epsilon, criterion)

# ########## Instanciate PGD Attack ##########
pgd = Attacks.ProjectedGradientDescent(conv_model, eps=.1, alpha=.1, num_iter=10, random=True, criterion=criterion)

"""
# ----------------------------------------------------------------------
#			Evaluation : a - without Attack, b - under FGSM Attack, c - under PGD Attack
# ----------------------------------------------------------------------
"""
# a - #############	
loss_wo_attack, acc_wo_attack = NN.eval_model(conv_model_vs_FGM, test_loader,criterion)
# b - #############
loss_fgsm_attack, acc_fgsm_attack, fc_fgsm_delta = NN.eval_model(conv_model_vs_FGM, test_loader,criterion, fgsm)
# c - #############
loss_pgd_attack, accuracy_pgd_attack, cnn_pgd_delta = NN.eval_model(conv_model_vs_FGM, test_loader, criterion,pgd)
print("\n --- Conv2D FGSM Resilient Model ---\n")
print(f"\t Test Evaluation - Model Trained against FGSM Attack | Loss: {loss_wo_attack:4f} \t | Accuracy: {acc_wo_attack:2%}" )
print(f"\t Test Evaluation - After FGSM Attack |  Loss: {loss_fgsm_attack:4f} \t | Accuracy: {acc_fgsm_attack:2%} | Accuracy Gap FGSM: {(acc_fgsm_attack - acc_wo_attack):.2%}")
print(f"\t Test Evaluation - After PGD Attack |  Loss: {loss_pgd_attack:4f} \t | Accuracy: {accuracy_pgd_attack:2%} | Accuracy Gap PGD: {(accuracy_pgd_attack - acc_wo_attack):.2%}")


#print("\n 3.1 - ////////////////  Training Linear Fully Connected NN and Conv2D NN - Under PGD Attack for more resilience ------------------\n")
print(
"""
# ######################################################################################
#
#  Step 3.1
#           - If trained against PGD versions of the NN exist
#               => instanciate and load them
#
#           - Otherwise .
#                => instanciate, train against PGD attack and save them
# 
# ######################################################################################
""")

if path.exists(saved_models_path+'fc_model_vs_PGD.pt') and path.exists(saved_models_path+'conv_model_vs_PGD.pt') :
	"""
	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	#
	#						If exist, load robust FC Linear Model
	#
	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	"""	

	fc_model_vs_PGD=NN.FullyConnectedModel(input_dim = 784,
                          output_dim = 10,
                          hidden_dim =100)
	
	if torch.cuda.is_available() == False:
		fc_model_vs_PGD.load_state_dict(torch.load(saved_models_path+'fc_model_vs_PGD.pt', map_location=torch.device('cpu')))
	else : fc_model_vs_PGD.load_state_dict(torch.load(saved_models_path+'fc_model_vs_PGD.pt'))
	fc_model_vs_PGD.to(device)
	fc_model_vs_PGD.eval()
	print('\n --- Info  FC_model_vs_PGD already trained \n\n\t%s'%fc_model_vs_PGD)
    
	"""
	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	#
	#						If exist, load robust CONV2D Model
	#
	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	"""	

	conv_model_vs_PGD=NN.ConvModel(input_dim = 28,
                          output_dim = 10,
                          hidden_dim =64,
                          num_filters = 32,
                          kernel_size = 5,
                          stride = 1,
                          padding=2,
                          pool_size=2)
	if torch.cuda.is_available() == False:
		conv_model_vs_PGD.load_state_dict(torch.load(saved_models_path+'conv_model_vs_PGD.pt', map_location=torch.device('cpu')))
	else : conv_model_vs_PGD.load_state_dict(torch.load(saved_models_path+'conv_model_vs_PGD.pt'))
	conv_model_vs_PGD.to(device)
	conv_model_vs_PGD.eval()
	print('\n --- Info  Conv_model_vs_PGD already trained -------\n\t\t%s'%str(conv_model_vs_PGD))
else:
	print("\n--------------- Training Fully Connected Linear Model - against PGD Attack for more resilience -----------------------\n")
	"""
	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	#
	#						If not exist, train FC Linear Model against PGD and save
	#
	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	"""        
	fc_model_vs_PGD = NN.FullyConnectedModel(input_dim = 784,
                              output_dim = 10,
                              hidden_dim =100).to(device)
    
	opt = optim.SGD(fc_model_vs_PGD.parameters(), lr=.01, momentum=0.9, weight_decay=1e-04) # define your optimizer
	criterion = nn.CrossEntropyLoss() # define your loss function
	NN.train_model(fc_model_vs_PGD.to(device), criterion, opt, train_loader, epochs = 10,attack=pgd)
	model_name = saved_models_path+"fc_model_vs_PGD.pt"
	torch.save(fc_model_vs_PGD.state_dict(), model_name)
    
	print("\n--------------- Training Conv2D - against PGD Attack for more resilience -----------------------\n")
	"""
	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	#
	#						If not exist, train CONV2D Model against PGD and save
	#
	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	"""   
	conv_model_vs_PGD = NN.ConvModel(input_dim = 28,
                              output_dim = 10,
                              hidden_dim =64,
                              num_filters = 32,
                              kernel_size = 5,
                              stride = 1,
                              padding=2,
                              pool_size=2)
	opt = optim.SGD(conv_model_vs_PGD.parameters(), lr=.01, momentum=0.9, weight_decay=1e-04)
	criterion = nn.CrossEntropyLoss() # define your loss function
	NN.train_model(conv_model_vs_PGD.to(device), criterion, opt, train_loader, epochs = 10,attack=pgd)
	model_name = saved_models_path+"conv_model_vs_PGD.pt"
	torch.save(conv_model_vs_PGD.state_dict(), model_name)
    
print(
"""
# #######################################################################################
#
#  Step 3.2
#           - Evaluate on the Test Set each NN trained against PGD attack 
#              a - Standard evaluation (no attack)
#              b - Evaluation when attacked by FGSM algo to
#                  check if a PGD trained model is robuste against other attacks
#              c - Evaluation when attacked by PGD algo 
#
#              
# #######################################################################################  
""" )

"""
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#
#					Evaluation of FC Linear Model
#
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
"""
"""
# ----------------------------------------------------------------------
#
#					Define FGSM Attack and PGD Attack
#						FGSM will be used to see if a model trained 
#						against PGD is ressilient against FGSM
#
# ----------------------------------------------------------------------
"""

# ########## Instanciate FGSM Attack ##########
fgsm = Attacks.FastGradientSignMethod(fc_model, epsilon, criterion)

# ########## Instanciate PGD Attack ##########
pgd = Attacks.ProjectedGradientDescent(fc_model, eps=.1, alpha=.1, num_iter=10, random=True, criterion=criterion)

print("\n 3.2 - //////////////// Resilient Models Evaluation - Before and After PGD Attacks ////////////////\n")
"""
# ----------------------------------------------------------------------
#			Evaluation : a - without Attack, b - under FGSM Attack, c - under PGD Attack
# ----------------------------------------------------------------------
"""
# a - #############	
loss_wo_attack, acc_wo_attack = NN.eval_model(fc_model_vs_PGD, test_loader,criterion)
# b - #############
loss_fgsm_attack, acc_fgsm_attack, fc_fgsm_delta = NN.eval_model(fc_model_vs_PGD, test_loader,criterion, fgsm)
# c - #############
loss_pgd_attack, accuracy_pgd_attack, fc_pgd_delta = NN.eval_model(fc_model_vs_PGD, test_loader,criterion, pgd)
print("\n --- FC FGSM Resilient Model ---\n")
print(f"\t Test Evaluation - Model Trained against PGD Attack | Loss: {loss_wo_attack:.4f} \t | Accuracy: {acc_wo_attack:.2%}")
print(f"\t Test Evaluation - After FGSM Attack |  Loss: {loss_fgsm_attack:.4f} \t | Accuracy: {acc_fgsm_attack:.2%} | Accuracy Gap FGSM: {(acc_fgsm_attack - acc_wo_attack):.2%}")
print(f"\t Test Evaluation - After PGD Attack |  Loss: {loss_pgd_attack:.4f} \t | Accuracy: {accuracy_pgd_attack:.2%} | Accuracy Gap PGD: {(accuracy_pgd_attack - acc_wo_attack):.2%}")




"""
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#
#					Evaluation of CONV2D Model
#
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
"""
"""
# ----------------------------------------------------------------------
#
#					Define FGSM Attack and PGD Attack
#						FGSM will be used to see if a model trained 
#						against PGD is ressilient against FGSM
#
# ----------------------------------------------------------------------
"""

# ########## Instanciate FGSM Attack ##########
fgsm = Attacks.FastGradientSignMethod(conv_model, epsilon, criterion)

# ########## Instanciate PGD Attack ##########
pgd = Attacks.ProjectedGradientDescent(conv_model, eps=.1, alpha=.1, num_iter=10, random=True, criterion=criterion)

"""
# ----------------------------------------------------------------------
#			Evaluation : a - without Attack, b - under FGSM Attack, c - under PGD Attack
# ----------------------------------------------------------------------
"""
# a - #############	
loss_wo_attack, acc_wo_attack = NN.eval_model(conv_model_vs_PGD, test_loader,criterion)
# b - #############	
loss_fgsm_attack, acc_fgsm_attack, fc_fgsm_delta = NN.eval_model(conv_model_vs_PGD, test_loader,criterion, fgsm)
# c - #############	
loss_pgd_attack, accuracy_pgd_attack, cnn_pgd_delta = NN.eval_model(conv_model_vs_PGD, test_loader, criterion,pgd)

print("\n --- Conv2D PGD Resilient Model ---\n")
print(f"\t Test Evaluation - Model Trained against PGD Attack | Loss: {loss_wo_attack:4f} \t | Accuracy: {acc_wo_attack:2%}" )
print(f"\t Test Evaluation - After FGSM Attack |  Loss: {loss_fgsm_attack:4f} \t | Accuracy: {acc_fgsm_attack:2%} | Accuracy Gap FGSM: {(acc_fgsm_attack - acc_wo_attack):.2%}")
print(f"\t Test Evaluation - After PGD Attack |  Loss: {loss_pgd_attack:4f} \t | Accuracy: {accuracy_pgd_attack:2%} | Accuracy Gap PGD: {(accuracy_pgd_attack - acc_wo_attack):.2%}")
