import numpy as np
import kmeans
import common
import naive_em
import em

# TODO: Your code here


# # toy data

# # running K-means
# X = np.loadtxt("toy_data.txt")
# for k in [1,2,3,4]:
#     low_cost = np.inf
#     for seed in [0,1,2,3,4]:
#         mixture, post = common.init(X,k,seed)
#         mixture, post, cost = kmeans.run(X,mixture,post)
#         if cost< low_cost:
#             low_cost = cost
        
#         common.plot(X,mixture,post,title = "KM: k = "+ str(k) + ", seed = "+ str(seed) +", cost = "+ str(cost))
        
#     print("KM: k = ",k,", lowest cost = ",low_cost)
    
# # running GMM
# X = np.loadtxt("toy_data.txt")
# for k in [1,2,3,4]:
#     max_like = -np.inf
#     max_bic = -np.inf
#     for seed in [0,1,2,3,4]:
#         mixture, post = common.init(X,k,seed)
#         mixture, post, loglike = naive_em.run(X,mixture,post)
#         if loglike> max_like:
#             max_like = loglike
#         bic = common.bic(X,mixture,loglike)
#         if bic>max_bic:
#             max_bic = bic
        
#         common.plot(X,mixture,post,title = "GMM: k = "+ str(k) + ", seed = "+ str(seed) +", loglikelihood = "+ str("%.2f"%loglike)+", BIC = "+ str("%.2f"%bic))
        
#     print("GMM: k = ",k,", Max loglikelihood = ",max_like,", Max BIC = ",max_bic)
    
# Netflix data
print("Running on Netflix data\n")
X = np.loadtxt("netflix_incomplete.txt")
for k in [1,12]:
    max_like = -np.inf
    for seed in [0,1,2,3,4]:
        mixture, post = common.init(X,k,seed)
        mixture, post, loglike = em.run(X,mixture,post)
        if loglike> max_like:
            max_like = loglike

        print("k = ",k,"seed = ",seed,"loglikelihood = ",loglike)
        # common.plot(X,mixture,post,title = "GMM: k = "+ str(k) + ", seed = "+ str(seed) +", loglikelihood = "+ str("%.2f"%max_like))
        
    print("GMM: k = ",k,", Max loglikelihood = ",max_like,"\n")
    
# Fill in matrix in Netflix data
k = 12
seed = 1
mixture, post = common.init(X,k,seed)
mixture, post, loglike = em.run(X,mixture,post)
X = np.loadtxt("netflix_incomplete.txt")
X_gold = np.loadtxt('netflix_complete.txt')
X_pred = em.fill_matrix(X,mixture)
print("k = ",k,"seed = ",seed,"rmse = ",common.rmse(X_gold, X_pred))
    



