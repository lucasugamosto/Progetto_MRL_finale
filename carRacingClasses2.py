import numpy as np
import random as rd
import time as tm
import gymnasium as gym
import pickle as pc
# import matplotlib.pyplot as plt

class carRacingClass2:
    
    def __init__(self, numEp, alp, eps, lam, gam, K):
        #Inizializzazione delle variabili utili al fine del gioco
        self.sizeSpace = 96                      #Dimensione dello spazio osservato lungo una direzione
        self.forwardDistance = 68                #Dimensione dello spazio osservato in avanti
        self.A = 5                               #Numero di azioni possibili
        self.actionVariable = 3                  #Dimensione del vettore indicante l'azione
        self.steering = np.array([-1, 1])        #-1 = sterzata a SX, 1 = sterzata a DX
        self.gas = np.array([0, 1])              #0 = non accelerare, 1 = accelerare
        self.breaking = np.array([0, 1])         #0 = non frenare, 1 = frenare
        
        #Se l'azione "gas", l'azione "breaking" e l'azione "steering" valgono
        #0 allora si sta considerando l'azione "do nothing"
        
        #Inizializzazione della matrice delle azioni
        self.actionMatrix = np.zeros([self.A, self.actionVariable])
        
        for i in range(self.A):
            #Prima azione della matrice è "do nothing" quindi [0, 0, 0]
            
            #Seconda azione della matrice è "steer left" quindi [-1, 0, 0]
            if (i == 1):
                self.actionMatrix[i, 0] = self.steering[0]
            
            #Terza azione della matrice è "steer right" quindi [1, 0, 0]
            elif (i == 2):
                self.actionMatrix[i, 0] = self.steering[1]
                
            #Quarta azione della matrice è "gas" quindi [0, 1, 0]
            elif (i == 3):
                self.actionMatrix[i, 1] = self.gas[1]
            
            #Quinta azione della matrice è "breaking" quindi [0, 0, 1]
            elif (i == 4):
                self.actionMatrix[i, 2] = self.breaking[1]
        
        #Inizializzazione dei parametri decisi e modificati dall'utente
        self.numEpisodes = numEp                 #Numero di episodi totali
        self.alpha = alp                         #Parametro applicato all'aggiornamento
        self.epsilon = eps                       #Parametro applicato all'aggiornamento
        self.Lambda = lam                        #Parametro applicato all'aggiornamento
        self.gamma = gam                         #Parametro applicato all'aggiornamento
        self.k = K
        
        self.epsUpdate = 0.000075                #Indica di quanto diminuire "epsilon" ad ogni episodio
        self.saveVariable = 25                   #Indica dopo ogni quanti episodi salvare le variabili utili
        
    #--------------------------------------------------------------------------
    
    def initStage(self, case):
        #Inizializzazione del vettore della stima della funzione qualità e del
        #vettore dei rewards
        self.S = (self.sizeSpace ** 2) * self.forwardDistance   #Dimensione delle variabili dello spazio di stato
        if (case == 0):
            #Nuova simulazione che richiede l'inizializzazione (per la 1° volta)
            #del vettore delle stime della funzione qualità e nel vettore dei
            #rewards totali          
            self.Q = np.random.randn(self.S, self.A)                #Inizializzazione casuale del vettore della stima della funzione qualità
            self.G = np.zeros([self.numEpisodes, 1])                #Inizializzazione nulla del vettore dei rewards
        elif (case == 1):
            #Si continua lo studio con un vettore Q già addestrato in precedenza
            #ed un vettore G già popolato di reward precedenti
            file_path_Q = r'C:\Users\quadr\Desktop\Assignment FINALE 2\Q.npy'
            self.Q = np.load(file_path_Q)
            file_path_G = r'C:\Users\quadr\Desktop\Assignment FINALE 2\G.npy'
            self.G = np.load(file_path_G)
            #Si carica anche il valore di Epsilon a cui si era arrivati in
            #precedenza
            file_path_eps = r'C:\Users\quadr\Desktop\Assignment FINALE 2\eps.pkl'
            with open(file_path_eps, 'rb') as file:
                self.epsilon = pc.load(file)
    #--------------------------------------------------------------------------
    
    def convert_RGB_GrayScale(self, image):
        #Conversione dello spazio osservato dalla rappresentazione a colori RGB
        #in rappresentazione a scala di grigi
        
        #Si consideta la riga associata al baricentro dell'auto (Riga 67)
        if (self.k == 2):               
            #Per risparmiare costo computazionale e tempo si considera solo la 
            #riga utile al calcolo, quindi non si converte tutta l'immagine
            self.grayMatrix = np.zeros([2, self.sizeSpace])   #Utilizzato per il calcolo della distanza orizzontale (1° riga)
                                                              #e della distanza verticale (2° riga)
            for j in range(self.sizeSpace):
                red = image[67, j, 0]
                green = image[67, j, 1]
                blue = image[67, j, 2]
                
                #Riempimento del vettore in scala di grigi
                grayScale = (0.2989 * red) + (0.5870 * green) + (0.1140 * blue) #Calcolo del valore di luminanza
                self.grayMatrix[0, j] = grayScale                               #Componente del vettore in scala di grigi
            
            #Per calcolare la distanza in avanti dell'auto, si converte in 
            #scala di grigi la colonna 48, dalla riga 0 alla riga 67 (dove si
            #trova la punta dell'auto)
            for j in range(self.forwardDistance):
                red = image[j, 48, 0]
                green = image[j, 48, 1]
                blue = image[j, 48, 2]
            
                grayScale = (0.2989 * red) + (0.5870 * green) + (0.1140 * blue) #Calcolo del valore di luminanza
                self.grayMatrix[1, j] = grayScale                               #Componente del vettore in scala di grigi
        
    #--------------------------------------------------------------------------
    
    def epsGreedy(self, state):
        #Algoritmo per la scelta dell'azione da prendere in ogni singola
        #iterazione
        randomProb = np.random.rand()                        #Generazione di un numero casuale di probabilità tra 0 ed 1
        
        if (randomProb < self.epsilon):
            #Caso in cui si prende un'azione casualmente
            indAstar = rd.randint(0, self.A-1)                 #Scelta randomica dell'azione, prendendo un indice a caso
            Astar = self.actionMatrix[indAstar, :]             #Azione presa definitivamente
        else:
            #Caso in cui si prende l'azione con valore di stima della funzione
            #qualità più alta
            indAstar = np.where(self.Q == np.max(self.Q[state, :]))[1][0]      #Si cerca l'indice in Q con valore massimo
            Astar = self.actionMatrix[indAstar, :]                             #Azione presa definitivamente       
        return([Astar,indAstar])
    
    #--------------------------------------------------------------------------
    
    def distanceCalculation(self):
        #Riceve la matrice in scala di grigi (Vettore se si considera una sola
        #riga dell'immagine di partenza) e calcola la distanza dell'auto dai
        #bordi laterali
        if (self.k == 2):
            for j in range(self.sizeSpace):
                if (self.grayMatrix[0, j] < 110):
                    x_SX = j           #Posizione del bordo pista sinistro
                    break              #Uscita anticipata dal ciclo
            for j in range(self.sizeSpace - 1, 0, -1):
                if (self.grayMatrix[0, j] < 110):
                    x_DX = j           #Posizione del bordo pista destro
                    break              #Uscita anticipata dal ciclo
            for j in range(self.forwardDistance):
                if (self.grayMatrix[1,j] < 110):
                    x_FD = j           #Posizione del bordo pista avanti
                    break              #Uscita anticipata dal ciclo
                    
            #Calcolo delle distanze tra bordo pista e bordo auto
            distance_SX = 46 - x_SX    #46 = bordo SX dell'auto
            distance_DX = x_DX - 49    #49 = bordo DX dell'auto
            distance_FD = 67 - x_FD    #67 = bordo anteriore dell'auto
            
            #Normalizzazione delle distanze per avere un indice compreso tra 0
            #e (sizeMatrix - 1)
            normDistance_SX = distance_SX + 49 
            normDistance_DX = distance_DX + 49 
            normDistance_FD = distance_FD
            
        return([normDistance_SX, normDistance_DX, normDistance_FD])
    
    #--------------------------------------------------------------------------
    
    def convertCoordinate(self, distanceVector):
        #Funzione che prende in ingresso il vettore delle distanze tra l'auto e
        #i bordi della strada e restituisce lo stato corrispondente sulla 
        #matrice delle stime della funzione qualità Q
        state = np.ravel_multi_index((distanceVector[0], distanceVector[1], distanceVector[2]), (self.sizeSpace, self.sizeSpace, self.forwardDistance))
        return(state)
    
    #--------------------------------------------------------------------------        
    
    def SARSALambda(self, env, case):
        #Algoritmo per l'addestramento del gioco
        self.stableIteration = 50                     #n° di iterazioni utili alla stabilizzazione dello zoom e dell'ambiente
        
        if (case == 0):
            #Nuova simulazione che richiede di partire dall'episodio 0
            initialEpisode = 0
        elif (case == 1):
            #Continuare lo studio dall'episodio in cui ci si è fermati prima,
            #quindi si carica l'episodio salvato
            file_path_e = r'C:\Users\quadr\Desktop\Assignment FINALE 2\e.pkl'
            with open(file_path_e, 'rb') as file:
                initialEpisode = pc.load(file) + 1
        
        for e in range(initialEpisode, self.numEpisodes):
            start = tm.time()                         #Inizio calcolo del tempo necessario ad analizzare un singolo episodio
            
            if (e >= self.numEpisodes - 25):          #Numero di episodi dopo il quale rappresentare graficamente la mappa di gioco
                env = gym.make("CarRacing-v2", render_mode = "human")
            
            observation, info = env.reset(seed = 1)   #Inizializzazione dell'ambiente 2D ad ogni episodio

            for i in range(self.stableIteration + 2):
                if (i <= self.stableIteration):
                    #Inizialmente l'ambiente deve stabilizzarsi e zoomare lo 
                    #spazio osservabile, quindi la macchina per questi istanti
                    #di tempo esegue l'azione "do nothing"
                    env.action_space = self.actionMatrix[0, :]
                    action = env.action_space
                    observation, reward, terminated, truncated, info = env.step(action)
                
                    if (i == self.stableIteration):
                        #L'ambiente si è stabilizzato quindi si calcola lo
                        #stato di partenza
                        stableObservation = observation
                        carRacingClass2.convert_RGB_GrayScale(self, stableObservation)
                        distanceVector = carRacingClass2.distanceCalculation(self)
                        
                else:
                    #Una volta stabilizzato lo spazio osservato, ci troviamo
                    #nell'iterazione successiva e può iniziare l'episodio
                    
                    #Diminuzione del valore di Epsilon ad ogni episodio, per 
                    #ridurre l'esplorazione ed aumentare lo sfruttamento dei 
                    #valori noti
                    if (self.epsilon > 0.05):
                        self.epsilon = self.epsilon - self.epsUpdate
                    
                    #Passaggio dal vettore delle distanze al corrispettivo 
                    #indice della matrice della stima della funzione qualità
                    state = carRacingClass2.convertCoordinate(self, distanceVector)
                    
                    self.E = np.zeros([self.S, self.A])     #Inizializzazione della variabile utile all'apprendimento
                    #Calcolo dell'azione da prendere utilizzando l'algoritmo di
                    #Epsilon Greedy
                    [a, ind_a] = carRacingClass2.epsGreedy(self, state)
                    
                    #Contatore delle iterazioni all'interno del ciclo while
                    count = 0
                    
                    while (terminated == False and truncated == False):
                        #Finchè non si sarà raggiunto uno stato terminale o non
                        #sarà finito il numero massimo di iterazioni possibili
                        #si considerano le seguenti istruzioni
                        env.action_space = a
                        action = env.action_space
                        #La funzione env.step() è utilizzata per far avanzare
                        #l'ambiente simulato in base all'azione scelta
                        observation, reward, terminated, truncated, info = env.step(action)
                        
                        #Immagine RGB catturata all'iterazione t
                        image = observation
                        #Calcoliamo la matrice in scala di grigi associata alla
                        #immagine appena ottenuta
                        carRacingClass2.convert_RGB_GrayScale(self, image)
                        #Calcolo della distanza tra l'auto ed i bordi laterali
                        #per la definizione dello stato successivo
                        distanceVector = carRacingClass2.distanceCalculation(self)
                        nextState = carRacingClass2.convertCoordinate(self, distanceVector)
                        
                        [next_a, next_ind_a] = carRacingClass2.epsGreedy(self, nextState)
                        
                        #Aggiornamento del parametro utile succcessivamente per
                        #aggiornare la stima della funzione qualità
                        self.delta = reward + (self.gamma * self.Q[nextState, next_ind_a]) - self.Q[state, ind_a]
                        
                        #Aggiornamento del vettore dei rewards in cui ogni riga
                        #indica un episodio ed il valore in essa indica la
                        #ricompensa totale ottenuto nello specifico episodio
                        self.G[e, 0] = self.G[e, 0] + reward
                        
                        #Aggiornamento della "dutch trace" associata alla sola
                        #coppia stato - azione interessata
                        self.E[state, ind_a] = ((1 - self.alpha) * self.E[state, ind_a]) + 1
                        
                        #Aggiornamento della matrice di stima della funzione
                        #qualità
                        self.Q = self.Q + (self.alpha * self.delta * self.E)
                        
                        #Aggiornamento della "dutch trace" totale
                        self.E = self.gamma * self.Lambda * self.E
                        
                        #Le nuove variabili di stato e azione vengono salvate
                        #come quelle precedenti per eseguire tutti i passaggi
                        #nell'iterazione successiva
                        state = nextState
                        a = next_a
                        ind_a = next_ind_a
                        
                        #Aggiornamento del contatore delle iterazioni
                        count = count + 1
                        #print("iterazione n°:", count)
                        
                    if (e % self.saveVariable == 0):        #Episodio nel quale si salvano i dati raccolti
                        print("Salvataggio dei dati fino all'episodio:", e)
                        #Salvataggio del vettore delle stime Q
                        file_path_Q = r'C:\Users\quadr\Desktop\Assignment FINALE 2\Q.npy'
                        np.save(file_path_Q, self.Q)
                        #Salvataggio del vettore dei reward G
                        file_path_G = r'C:\Users\quadr\Desktop\Assignment FINALE 2\G.npy'
                        np.save(file_path_G, self.G)
                        #Salvataggio del parametro Epsilon (probabilità di scelta dell'azione)
                        file_path_eps = r'C:\Users\quadr\Desktop\Assignment FINALE 2\eps.pkl'
                        with open(file_path_eps, 'wb') as file:
                            pc.dump(self.epsilon, file)                 
                        #Salvataggio del parametro e (indice dell'episodio corrente)
                        file_path_e = r'C:\Users\quadr\Desktop\Assignment FINALE 2\e.pkl'
                        with open(file_path_e, 'wb') as file:
                            pc.dump(e, file)
                
            finish = tm.time()
            totalTime = finish - start
            print("Tempo per l'episodio ", e, ":", totalTime, "\n")