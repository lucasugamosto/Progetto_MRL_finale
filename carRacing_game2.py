import gymnasium as gym
import carRacingClasses2 as CRC2

#Il parametro 'render_mode = "human"' permette di rappresentare graficamente
#lo spazio di lavoro; se questo non si vuole visualizzare si esegue la seguente
#istruzione 'render_mode = None'
# env = gym.make("CarRacing-v2", render_mode = "human")
env = gym.make("CarRacing-v2", render_mode = None)

#Inizializzazione della classe in cui sono definite le variabili e le funzioni
#Primo input = numero di episodi
#Secondo input = alpha (Deve avere valore tale da rispettare le condizioni di convergenza => Alpha -> 0)
#Terzo input = epsilon (Epsilon -> 0 => si sfrutta la conoscenza per la scelta dell'azione | Epsilon -> 1 => si esplorano le azioni sconosciute)
#Quarto input = lambda (Lambda -> 0 => si ottiene un MC algorithm | Lambda -> 1 => si ottiene un one-step TD algorithm)
#Quinto input = gamma (Gamma -> 0 => importanza data ai reward immediati | Gamma -> 1 => importanza data ai reward futuri)
#Sesto input = k

numEpisodes = 10000
Alpha = 0.005
initialEpsilon = 0.8
Lambda = 0.8
Gamma = 0.5
k = 2
Class = CRC2.carRacingClass2(numEpisodes, Alpha, initialEpsilon, Lambda, Gamma, k)

#Inizializzazione delle variabili utili alla fase di aggiornamento.
#Se 1° parametro in input vale 0 allora si inizial una nuova simulazione,
#se invece vale 1 allora si continua una vecchia simulazione
Class.initStage(1)

#Esecuzione dell'algoritmo di apprendimento.
#Se 2° parametro in input vale 0 allora si inizial una nuova simulazione, 
#se invece vale 1 allora si continua una vecchia simulazione
Class.SARSALambda(env, 1)

env.close()

