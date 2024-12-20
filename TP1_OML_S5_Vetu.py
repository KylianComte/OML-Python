#########################################################################
#Exemple 2
#######################################################################

import math
import matplotlib.pyplot as plt
import numpy as np

from statistics import variance


# calcul d'une moyenne et d'une variance
nombre_enfants=[0,1,2,3,4,5,6,7,8]
nombre_salariés=[241,1398,516,129,34,22,2,2,1]
effectif_total=sum(nombre_salariés)

moyenne=0
variance=0
somme=0
som_carre=0
Ne=len(nombre_enfants)

for i in range(Ne):
  # remplir ici code pour calculer la moyenne et la variance
  moyenne+=nombre_enfants[i]*nombre_salariés[i]/effectif_total
  variance+=(nombre_enfants[i]**2)*nombre_salariés[i]/effectif_total
  ecart_type=math.sqrt(variance)


# fin code



# utilisation de la fonction average de numpy et comparaison avec votre résultat
nombre_moyen_enfants=np.average(nombre_enfants,weights=nombre_salariés)

# impression des résultats
# ecrire votre code ici

print('moyenne =',moyenne)
print('moyenne numpy =',nombre_moyen_enfants)
print('variance =',variance)
print('ecart-type =',ecart_type)

# fin de code


################################################################################
#Exercice 1
###############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Définition de la loi normale
def normal(x,u,sigma):
  return (1/(sigma*np.sqrt(2*np.pi))) *np.exp( - np.power( (x- u) ,2)/(2*np.power(sigma,2)))


# Lecture des données AII
donnees_AII=np.array(pd.read_excel('notes_S4.xlsx',sheet_name="AII"))
anglais_AII=donnees_AII[0:46,0]
maths_AII=donnees_AII[0:46,1]
PPP_AII=donnees_AII[0:46,2]

plt.figure(1)
# Histogramme des notes 
plt.hist(anglais_AII,[0,2,4,6,8,10,12,14,16,18,20],color='blue',edgecolor='black',density=True)
plt.title('notes anglais AII')
plt.xlabel('Notes')
plt.ylabel('Fréquences')

#Calcul de la moyenne
moyenne_ang=np.mean(anglais_AII)
print('moyenne anglais AII=',moyenne_ang)

# Calcul de l'écart type
ecart_type_ang=np.std(anglais_AII)
print('ecart anglais AII=',ecart_type_ang)

# Tracé de la loi normale
abscisses_notes=np.linspace(0,20,1000)
plt.plot(abscisses_notes, normal(abscisses_notes,moyenne_ang,ecart_type_ang),'r')
axes=plt.gca()
y_min_plot,y_max_plot=axes.get_ylim()
print(y_max_plot)
plt.axvline(moyenne_ang, ymin=0, ymax=normal(moyenne_ang,moyenne_ang,ecart_type_ang)/y_max_plot, color="red") 
plt.text(5, 0.1, "moyenne= {0:6.3f}".format(moyenne_ang),color="red")
plt.axvline(moyenne_ang-ecart_type_ang, ymin=0, ymax=normal(moyenne_ang,moyenne_ang,ecart_type_ang)/y_max_plot, color="red")
plt.axvline(moyenne_ang+ecart_type_ang, ymin=0, ymax=normal(moyenne_ang,moyenne_ang,ecart_type_ang)/y_max_plot, color="red")


plt.figure(2)

# ecrire le code pour la moyenne de maths, 3PE

#maths

plt.hist(maths_AII,[0,2,4,6,8,10,12,14,16,18,20],color='blue',edgecolor='black',density=True)
plt.title('notes maths AII')
plt.xlabel('Notes')
plt.ylabel('Fréquences')

#Calcul de la moyenne
moyenne_mat=np.mean(maths_AII)
print('moyenne maths AII=',moyenne_mat)

# Calcul de l'écart type
ecart_type_mat=np.std(maths_AII)
print('ecart maths AII=',ecart_type_mat)

# Tracé de la loi normale
abscisses_notes=np.linspace(0,20,1000)
plt.plot(abscisses_notes, normal(abscisses_notes,moyenne_mat,ecart_type_mat),'r')
axes=plt.gca()
y_min_plot,y_max_plot=axes.get_ylim()
print(y_max_plot)
plt.axvline(moyenne_mat, ymin=0, ymax=normal(moyenne_mat,moyenne_mat,ecart_type_mat)/y_max_plot, color="red") 
plt.text(5, 0.1, "moyenne= {0:6.3f}".format(moyenne_mat),color="red")
plt.axvline(moyenne_mat-ecart_type_mat, ymin=0, ymax=normal(moyenne_mat,moyenne_mat,ecart_type_mat)/y_max_plot, color="red")
plt.axvline(moyenne_mat+ecart_type_mat, ymin=0, ymax=normal(moyenne_mat,moyenne_mat,ecart_type_mat)/y_max_plot, color="red")


plt.figure(3)


#PPP
plt.hist(PPP_AII,[0,2,4,6,8,10,12,14,16,18,20],color='blue',edgecolor='black',density=True)
plt.title('notes PPP AII')
plt.xlabel('Notes')
plt.ylabel('Fréquences')

#Calcul de la moyenne
moyenne_PPP=np.mean(PPP_AII)
print('moyenne PPP AII=',moyenne_PPP)

# Calcul de l'écart type
ecart_type_PPP=np.std(PPP_AII)
print('ecart PPP AII=',ecart_type_PPP)

# Tracé de la loi normale
abscisses_notes=np.linspace(0,20,1000)
plt.plot(abscisses_notes, normal(abscisses_notes,moyenne_PPP,ecart_type_PPP),'r')
axes=plt.gca()
y_min_plot,y_max_plot=axes.get_ylim()
print(y_max_plot)
plt.axvline(moyenne_PPP, ymin=0, ymax=normal(moyenne_PPP,moyenne_PPP,ecart_type_PPP)/y_max_plot, color="red") 
plt.text(5, 0.1, "moyenne= {0:6.3f}".format(moyenne_PPP),color="red")
plt.axvline(moyenne_PPP-ecart_type_PPP, ymin=0, ymax=normal(moyenne_PPP,moyenne_PPP,ecart_type_PPP)/y_max_plot, color="red")
plt.axvline(moyenne_PPP+ecart_type_PPP, ymin=0, ymax=normal(moyenne_PPP,moyenne_PPP,ecart_type_PPP)/y_max_plot, color="red")

# Lecture des données ESE
donnees_ESE=np.array(pd.read_excel('notes_S4.xlsx',sheet_name="ESE"))
anglais_ESE=donnees_ESE[0:46,0]
maths_ESE=donnees_ESE[0:46,1]
PPP_ESE=donnees_ESE[0:46,2]

plt.figure(4)
# Histogramme des notes 
plt.hist(anglais_ESE,[0,2,4,6,8,10,12,14,16,18,20],color='blue',edgecolor='black',density=True)
plt.title('notes anglais ESE')
plt.xlabel('Notes')
plt.ylabel('Fréquences')

#Calcul de la moyenne
moyenne_ang=np.mean(anglais_ESE)
print('moyenne anglais ESE=',moyenne_ang)

# Calcul de l'écart type
ecart_type_ang=np.std(anglais_ESE)
print('ecart anglais ESE=',ecart_type_ang)

# Tracé de la loi normale
abscisses_notes=np.linspace(0,20,1000)
plt.plot(abscisses_notes, normal(abscisses_notes,moyenne_ang,ecart_type_ang),'r')
axes=plt.gca()
y_min_plot,y_max_plot=axes.get_ylim()
print(y_max_plot)
plt.axvline(moyenne_ang, ymin=0, ymax=normal(moyenne_ang,moyenne_ang,ecart_type_ang)/y_max_plot, color="red") 
plt.text(5, 0.1, "moyenne= {0:6.3f}".format(moyenne_ang),color="red")
plt.axvline(moyenne_ang-ecart_type_ang, ymin=0, ymax=normal(moyenne_ang,moyenne_ang,ecart_type_ang)/y_max_plot, color="red")
plt.axvline(moyenne_ang+ecart_type_ang, ymin=0, ymax=normal(moyenne_ang,moyenne_ang,ecart_type_ang)/y_max_plot, color="red")


plt.figure(5)

# ecrire le code pour la moyenne de maths, 3PE

#maths

plt.hist(maths_ESE,[0,2,4,6,8,10,12,14,16,18,20],color='blue',edgecolor='black',density=True)
plt.title('notes maths ESE')
plt.xlabel('Notes')
plt.ylabel('Fréquences')

#Calcul de la moyenne
moyenne_mat=np.mean(maths_ESE)
print('moyenne maths ESE=',moyenne_mat)

# Calcul de l'écart type
ecart_type_mat=np.std(maths_ESE)
print('ecart maths ESE=',ecart_type_mat)

# Tracé de la loi normale
abscisses_notes=np.linspace(0,20,1000)
plt.plot(abscisses_notes, normal(abscisses_notes,moyenne_mat,ecart_type_mat),'r')
axes=plt.gca()
y_min_plot,y_max_plot=axes.get_ylim()
print(y_max_plot)
plt.axvline(moyenne_mat, ymin=0, ymax=normal(moyenne_mat,moyenne_mat,ecart_type_mat)/y_max_plot, color="red") 
plt.text(5, 0.1, "moyenne= {0:6.3f}".format(moyenne_mat),color="red")
plt.axvline(moyenne_mat-ecart_type_mat, ymin=0, ymax=normal(moyenne_mat,moyenne_mat,ecart_type_mat)/y_max_plot, color="red")
plt.axvline(moyenne_mat+ecart_type_mat, ymin=0, ymax=normal(moyenne_mat,moyenne_mat,ecart_type_mat)/y_max_plot, color="red")


plt.figure(6)


#PPP
plt.hist(PPP_ESE,[0,2,4,6,8,10,12,14,16,18,20],color='blue',edgecolor='black',density=True)
plt.title('notes PPP ESE')
plt.xlabel('Notes')
plt.ylabel('Fréquences')

#Calcul de la moyenne
moyenne_PPP=np.mean(PPP_ESE)
print('moyenne PPP ESE=',moyenne_PPP)

# Calcul de l'écart type
ecart_type_PPP=np.std(PPP_ESE)
print('ecart PPP ESE=',ecart_type_PPP)

# Tracé de la loi normale
abscisses_notes=np.linspace(0,20,1000)
plt.plot(abscisses_notes, normal(abscisses_notes,moyenne_PPP,ecart_type_PPP),'r')
axes=plt.gca()
y_min_plot,y_max_plot=axes.get_ylim()
print(y_max_plot)
plt.axvline(moyenne_PPP, ymin=0, ymax=normal(moyenne_PPP,moyenne_PPP,ecart_type_PPP)/y_max_plot, color="red") 
plt.text(5, 0.1, "moyenne= {0:6.3f}".format(moyenne_PPP),color="red")
plt.axvline(moyenne_PPP-ecart_type_PPP, ymin=0, ymax=normal(moyenne_PPP,moyenne_PPP,ecart_type_PPP)/y_max_plot, color="red")
plt.axvline(moyenne_PPP+ecart_type_PPP, ymin=0, ymax=normal(moyenne_PPP,moyenne_PPP,ecart_type_PPP)/y_max_plot, color="red")





# fin de code
plt.show()

###############################################################
#Exercice 2
#############################################################
import matplotlib.pyplot as plt
import numpy as np


#tracé de l'histogramme loi normale et densite de proba
# donner la valeur de l'écart-type
ecart_type= 10
plt.figure(7)
nbSimu=1000
Simu=np.random.normal(100,ecart_type,size=nbSimu)
#Simu=np.random.normal(100, 30) +30+5*np.random.randint(0, 256)
#formule desité de probabilité loi normale
def normal(x,u,sigma):
  return (1/(sigma*np.sqrt(2*np.pi))) *np.exp( - np.power( (x- u) ,2)/(2*np.power(sigma,2)))

plt.hist(Simu,bins=15,density=True,edgecolor="k");
x=np.linspace(10,200,2000)
plt.plot(x, normal(x,100,ecart_type),'r')
plt.title("simulation loi normale avec écart-type = {0:6.3f}".format(ecart_type))

ecart_type= 20
plt.figure(8)
nbSimu=1000
Simu=np.random.normal(100,ecart_type,size=nbSimu)
#Simu=np.random.normal(100, 30) +30+5*np.random.randint(0, 256)
#formule desité de probabilité loi normale
def normal(x,u,sigma):
  return (1/(sigma*np.sqrt(2*np.pi))) *np.exp( - np.power( (x- u) ,2)/(2*np.power(sigma,2)))

plt.hist(Simu,bins=15,density=True,edgecolor="k");
x=np.linspace(10,200,2000)
plt.plot(x, normal(x,100,ecart_type),'r')
plt.title("simulation loi normale avec écart-type = {0:6.3f}".format(ecart_type))

ecart_type= 50
plt.figure(9)
nbSimu=1000
Simu=np.random.normal(100,ecart_type,size=nbSimu)
#Simu=np.random.normal(100, 30) +30+5*np.random.randint(0, 256)
#formule desité de probabilité loi normale
def normal(x,u,sigma):
  return (1/(sigma*np.sqrt(2*np.pi))) *np.exp( - np.power( (x- u) ,2)/(2*np.power(sigma,2)))

plt.hist(Simu,bins=15,density=True,edgecolor="k");
x=np.linspace(10,200,2000)
plt.plot(x, normal(x,100,ecart_type),'r')
plt.title("simulation loi normale avec écart-type = {0:6.3f}".format(ecart_type))


plt.show()
