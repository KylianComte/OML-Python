#########################################################
# TP2 : series statistiques doubles
########################################################

import math
import matplotlib.pyplot as plt
import numpy as np

from statistics import fmean, variance
###############################################
# Exercice 1 :Première méthode avec la matrice de covariance
###############################################
temperature=np.array([3.0, 5.3, 8.4, 13.1, 14.1, 18.7, 22.0, 20.4, 17.5, 13.1, 8.6, 4.0])
precipitations=np.array([74,61,61,28,68,115,28,25,45,83,144,60])

# Calcul de la covariance
matrice_cov=np.cov(temperature,precipitations,bias=True)
print('matrice de covariance=',matrice_cov)
print('la covariance de X et Y est =',matrice_cov[0][1])

# Calcul de la variance
matrice_var=variance(temperature)

# Tracé des points 
plt.figure(1)
plt.scatter(temperature,precipitations)
a1=matrice_cov[0][1]/matrice_var

xmoy=fmean(temperature)
ymoy=fmean(precipitations)
b1= ymoy-a1*xmoy


print('moyenneTemperature=',xmoy)
print('moyennePrecipitations=',ymoy)


# droite de régression
print('la droite de régression a pour équation y=',a1,'x+',b1)
x_trace=np.linspace(min(temperature),max(temperature),100)

#Tracé de la droite de régression avec un titre et les noms des axes
plt.plot(x_trace,a1*x_trace+b1,'red')
plt.title('températures et parécipitations dans le Sud Ouest en 2010')
plt.xlabel('Températures en °C')
plt.ylabel('Précipitation en mm')
plt.show()


###################################################################
# Exercice 1 :Deuxième méthode : utilisation de la pseudo inverse
##########################################################

# Création de la matrice N
N = np.vstack([temperature, np.ones(len(temperature))]).T

# Calcul de la pseudoinverse en utilisant np.linalg.pinv
a2,b2=np.linalg.pinv(N).dot(precipitations)
print('la droite de régression a pour équation y=',a2,'x+',b2)

# Tracé des points
plt.figure(2)
plt.scatter(temperature,precipitations)
x_trace=np.linspace(min(temperature),max(temperature),100)
plt.plot(x_trace,a2*x_trace+b2,'red')
plt.title('températures et parécipitations dans le Sud Ouest en 2010')
plt.xlabel('Températures en °C')
plt.ylabel('Précipitation en mm')
plt.show()

###################################################################
# Exercice 1 :Troisième méthode : utilisation de la fonction np.linalg.lstsq de Python
##########################################################
a3,b3=np.linalg.lstsq(N,precipitations,rcond=None)[0]
print('la droite de régression a pour équation y=',a3,'x+',b3)

# Tracé des points
plt.figure(3)
plt.scatter(temperature,precipitations)
x_trace=np.linspace(min(temperature),max(temperature),100)
plt.plot(x_trace,a3*x_trace+b3,'red')
plt.title('températures et parécipitations dans le Sud Ouest en 2010')
plt.xlabel('Températures en °C')
plt.ylabel('Précipitation en mm')
plt.show()

#####################
# Exercice 2
#A compléter



#####################
# Exercice 3
rangAnnee=np.array([1,2,3,5])
chiffreAffaires=np.array([645,700,840,1235])

# Création de la matrice N
N = np.vstack([rangAnnee, np.ones(len(rangAnnee))]).T

# Calcul de la pseudoinverse en utilisant np.linalg.pinv
a4,b4=np.linalg.pinv(N).dot(chiffreAffaires)
print('la droite de régression a pour équation y=',a4,'x+',b4)

# Estimer 2018 
chiffreAffaires2018=a4*4+b4
print('le chiffre d affaires de 2018 est =',chiffreAffaires2018)

# Tracé des points
plt.figure(4)
plt.scatter(rangAnnee,chiffreAffaires)
x_trace=np.linspace(min(rangAnnee),max(rangAnnee),100)
plt.plot(x_trace,a4*x_trace+b4,'green')
plt.title('Rang et Chiffre affaires')
plt.xlabel('Rang')
plt.ylabel('Chifre d affaires en milliers d euros')
plt.show()
