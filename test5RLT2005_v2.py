import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import random as r
import matplotlib.animation as an
from   scipy.optimize    import root 
# Fonction pour simuler le modèle de Lotka-Volterra
def lotka_volterra(prey_init, predator_init, alpha, beta, delta, gamma, time):
    prey = np.zeros(time)
    predator = np.zeros(time)
    prey[0] = prey_init
    predator[0] = predator_init

    for t in range(1, time):
        prey[t] = prey[t-1] + (alpha * prey[t-1] - beta * prey[t-1] * predator[t-1])
        predator[t] = predator[t-1] + (delta * prey[t-1] * predator[t-1] - gamma * predator[t-1])

    return prey, predator

def init_grid(size, density, type1_ratio):
    grid = np.zeros((size, size), dtype=int)
    num_agents = int(size * size * density)

    num_type1 = int(num_agents * type1_ratio)
    num_type2 = num_agents - num_type1

    # Remplir la grille avec des agents
    for _ in range(num_type1):
        while True:
            x, y = r.randint(0, size-1), r.randint(0, size-1)
            if grid[x, y] == 0:
                grid[x, y] = 1  # Type 1
                break

    for _ in range(num_type2):
        while True:
            x, y = r.randint(0, size-1), r.randint(0, size-1)
            if grid[x, y] == 0:
                grid[x, y] = 2  # Type 2
                break

    return grid

# Fonction pour vérifier la satisfaction d'un agent
def is_satisfied(grid, x, y, tolerance):
    type_agent = grid[x, y]
    if type_agent == 0:
        return True  # Case vide

    neighbors = [(i, j) for i in range(max(0, x-1), min(x+2, grid.shape[0]))
                          for j in range(max(0, y-1), min(y+2, grid.shape[1]))
                          if (i, j) != (x, y)]

    same_type_count = sum(1 for i, j in neighbors if grid[i, j] == type_agent)
    return same_type_count / len(neighbors) >= tolerance

# Fonction pour animer les agents
def update(grid, tolerance):
    new_grid = grid.copy()
    empty_cells = [(i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1]) if grid[i, j] == 0]
    
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] != 0 and not is_satisfied(grid, i, j, tolerance):
                if empty_cells:
                    new_cell = r.choice(empty_cells)
                    new_grid[new_cell] = grid[i, j]
                    new_grid[i, j] = 0
                    empty_cells.remove(new_cell)
    
    return new_grid

# Fonction principale de l'application
def main():
    st.title("La modélisation de l'emploi - RLT2005")

    # Barre de navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choisissez un modèle", ["néoclassique","Modèle Proie-Prédateur", "Théorie des jeux", "Schelling"])

# Page du modèle néoclassique
    if page == "néoclassique":
        st.subheader("Simulation du modèle")
        # Paramètres modifiables
        dpi=300
        size  = 50
        K = st.sidebar.number_input("Stock de capital", value=20)
        A = st.sidebar.number_input("Niveau de la technologie", value=20)
        alpha = 0.7
        alpha_plot = 0.8
        D_a = st.sidebar.slider("Choc sur l'élasticité de la production au capital", -1.0, 1.0, 0.2)
        D_K = st.sidebar.slider("Choc sur le stock de capital", -50, 50, 20)
        D_A = st.sidebar.slider("Choc sur la technologie", -50, 50, 20)
        rW = np.linspace(1, size, size*2)  # Real wage
        
        T = st.sidebar.number_input("Stock d'heures disponibles", value=24)
        beta =  0.7  # Utility elasticity of consumption
        beta = st.sidebar.number_input("Élasticité de l'utilité à la consommation", value=0.7)
        I    = 50    # Non-labor income
        I = st.sidebar.number_input("Revenu hors emploi", value=50)
        rW2  = 25    # Real wage
        # Arrays
        L = np.linspace(1, T, T*4)  # Array of labor hours from 0 to T
        
        st.subheader("Graphiques du marché du travail")
        def Ndemand(A, K, rW, alpha):
            Nd = K * ((1 - alpha)*A/rW)**(1/alpha)
            return Nd

        Nd   = Ndemand(A    , K    , rW, alpha)      
        Nd_K = Ndemand(A    , K+D_K, rW, alpha)      
        Nd_A = Ndemand(A+D_A, K    , rW, alpha)      
        Nd_a = Ndemand(A    , K    , rW, alpha+D_a)

        # AXIS RANGE
        x = np.linspace(0, 50, 100)
        axis_range = [0, 50, 0, size]

        # BUILD PLOT
        plt.figure(figsize=(8, 8), dpi=dpi)
        # ADD DEMAND CURVES
        plt.plot(Nd  , rW, "k-", alpha=alpha_plot, label="Labor demand", linewidth=3)
        plt.plot(Nd_K, rW, "b-", alpha=alpha_plot, label="Capital shock")
        plt.plot(Nd_A, rW, "r-", alpha=alpha_plot, label="Productivity shock")
        plt.plot(Nd_a, rW, "g-", alpha=alpha_plot, label="Output elasticity of K shock")
        # AXIS
        plt.axhline(0, color='black', lw=0.5, ls='--')  # Axe horizontal
        plt.axvline(0, color='black', lw=0.5, ls='--')  # Axe vertical
        #SETTINGS
        plt.grid()
        plt.xlabel(r'Nd')
        plt.ylabel(r'w/P')
        plt.title("Demande de travail")
        plt.legend() 
        # Affichage du graphique dans Streamlit
        st.pyplot(plt)
        # Optionnel : Réinitialiser le graphique
        plt.close()  # Ferme la figure pour éviter les affichages multiples

        Ustar = (beta*(I + T*rW2))**beta * ((1 - beta)*(I + T*rW2)/rW2)**(1 - beta)
        Lstar = (1 - beta)*((I + T*rW2)/rW2)
        Cstar = beta*(I + T*rW2)

        # Consumption function
        def C_indiff(U, L, beta):
            C_indiff = (U/L**(1 - beta))**(1/beta)
            return C_indiff

        # Budget constraint
        def Budget(I, rW2, L):
            Budget = (I + T*rW2) - rW2*L
            return Budget
        y_max = 2*Budget(I, rW2, 0)

        # AXIS RANGE
        axis_range = [0, T+1, 0, y_max]
        # Populate the indifference curve and the budget constraint
        B = Budget(I, rW2, L)
        C = C_indiff(Ustar, L, beta)
        # BUILD PLOT
        plt.figure(figsize=(8, 8), dpi=dpi)
        # ADD LINES TO THE PLOT
        plt.plot(L, C, "g-", alpha=0.75, label="Indifference curve")
        plt.plot(L, B, "k-", alpha=0.75, label="Budget constraint")
        plt.axvline(x=T    , ymin=0, alpha=0.75, ymax=I/y_max, color='k') # Add I
        plt.axvline(x=Lstar, ymin=0, alpha=0.75, ymax = Cstar/y_max, ls=':', color='k') 
        plt.axhline(y=Cstar, xmin=0, alpha=0.75, xmax = Lstar/T    , ls=':', color='k') 
        plt.plot(Lstar, Cstar, 'bo')
        # LABELS
        plt.text(0.1      ,  Cstar+5, np.round(Cstar, 1), color="k")
        plt.text(Lstar+0.2,  10     , np.round(Lstar, 1), color="k")
        # SETTINGS
        plt.xlabel("Leisure")
        plt.ylabel("Real income")
        plt.title("Choix du ménage")
        plt.legend() 
        # Affichage du graphique dans Streamlit
        st.pyplot(plt)
        # Optionnel : Réinitialiser le graphique
        plt.close()  # Ferme la figure pour éviter les affichages multiples
        
        def Nsupply(rW, beta, I):
            Lsupply = T - (1 - beta)*((T*rW + I)/rW)
            return Lsupply

        D_I = 25     # Shock to non-income labor
        D_b = 0.10   # Shock to beta

        Ns   = Nsupply(rW, beta    , I)
        Ns_b = Nsupply(rW, beta+D_b, I)
        Ns_I = Nsupply(rW, beta    , I+D_I)

        y_max = np.max(Ns)
        axis_range = [0, T, 0, y_max]

        # BUILD PLOT
        plt.figure(figsize=(10, 8), dpi=dpi)
        # ADD LABOR SUPPLY LINES
        plt.plot(Ns  , rW, "k", alpha=0.75, label="Labor supply", linewidth=3)
        plt.plot(Ns_I, rW, "b", alpha=0.75, label="Non-labor income shock")
        plt.plot(Ns_b, rW, "r", alpha=0.75, label="Consumption elasticy of utility shock")
        # SETTINGS
        plt.xlabel("Heures de travail")
        plt.ylabel("salaire réel")
        plt.title("Offre de travail")
        plt.legend() 
        # Affichage du graphique dans Streamlit
        st.pyplot(plt)
        # Optionnel : Réinitialiser le graphique
        plt.close()  # Ferme la figure pour éviter les affichages multiples
        
      def Eq_Wage(rW):
            return Ndemand(A, K, rW, alpha) - Nsupply(rW, beta, I)

        rW_0 = 10
        sol = root(Eq_Wage, rW_0)

        w_star = float(sol.x[0])                    # salaire d'équilibre (scalaire)
        N_star = float(Nsupply(w_star, beta, I))    # emploi d'équilibre (scalaire)

        # 12|PLOT LABOR MARKET EQUILIBRIUM
        plt.figure(figsize=(8, 8), dpi=dpi)

        # Courbes
        plt.plot(Ns[1:T], rW[1:T], "k", label="Labor supply")
        plt.plot(Nd[1:T], rW[1:T], "k", label="Labor demand")

        # Point d'équilibre
        plt.plot(N_star, w_star, "bo")

        # Labels / annotations
        plt.text(4.5, 11, "Labor demand")
        plt.text(20, 5, "Labor supply")
        plt.text(0.2, w_star + 0.3, np.round(w_star, 1))
        plt.text(N_star + 0.3, 0.3, np.round(N_star, 1))

        # Optionnel: limites d'axes (si tu veux vraiment les fixer)
        # plt.xlim(0, T)
        # plt.ylim(0, 12)

        plt.xlabel("Heures de travail")
        plt.ylabel("salaire réel")
        plt.title("Équilibre")
        plt.legend()

        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)
        
        st.subheader("Crédits")
        st.write("Nicolas Cachanosky, Metropolitan State University of Denver ; ncachano@msudenver.edu ; http://www.ncachanosky.com")    
        # Page du modèle Lotka-Volterra
    elif page == "Modèle Proie-Prédateur":
        st.subheader("Simulation du modèle")

        # Paramètres modifiables
        prey_init = st.sidebar.number_input("Population initiale de proies", value=40)
        predator_init = st.sidebar.number_input("Population initiale de prédateurs", value=9)
        alpha = st.sidebar.number_input("Taux de croissance des proies (alpha)", value=0.1)
        beta = st.sidebar.number_input("Taux de prédation (beta)", value=0.02)
        delta = st.sidebar.number_input("Taux de reproduction des prédateurs (delta)", value=0.01)
        gamma = st.sidebar.number_input("Taux de mortalité des prédateurs (gamma)", value=0.1)
        time = st.sidebar.slider("Temps de simulation (en jours)", min_value=10, max_value=200, value=100)

        # Simulation
        prey, predator = lotka_volterra(prey_init, predator_init, alpha, beta, delta, gamma, time)

        # Affichage des résultats
        st.subheader("Graphique de la dynamique des populations")
        fig, ax = plt.subplots()
        ax.plot(prey, label='Proies', color='blue')
        ax.plot(predator, label='Prédateurs', color='red')
        ax.set_xlabel('Temps')
        ax.set_ylabel('Population')
        ax.legend()
        st.pyplot(fig)

    # Page 2 (exemple)
    elif page == "Théorie des jeux":
        st.subheader("Simulation du modèle")
        with st.expander("Voir les règles du jeu", expanded=True):
            st.markdown("""
       **Actions possibles**
        - **C** : coopérer
        - **T** : trahir

        **Gains par tour (joueur 1, joueur 2)**
        - (C, C) → (3, 3)
        - (C, T) → (0, 5)
        - (T, C) → (5, 0)
        - (T, T) → (1, 1)

        **Déroulement**
        - Deux stratégies s’affrontent pendant **N tours**.
        - Chaque stratégie choisit C ou T à chaque tour (certaines utilisent l’historique des coups).
        - On additionne les gains sur les N tours.

        **Tournoi**
        - Chaque stratégie joue contre toutes les autres (et contre elle-même dans ton code actuel).
        - On calcule :
          1) les **points totaux** (somme des gains)
          2) les **victoires de confrontation** (1 si un match est gagné, 0.5 si égalité).
            """)
        st.write("liste des personnages jouables")
        st.write("cooperatif : fais C à tous les tours")
        st.write("traitre : fais T à tous les tours")
        st.write("rancunier : fais C, mais si une fois trahi, il ne fera plus que T")
        st.write("copieur : fais ce qu'a fait son adversaire au tour d'avant")
        st.write("aleatoire : joue aléatoirement C ou T")
        st.write("vicieux : joue C, mais si son adversaire fais 3 C de suite, il trahi à la prochaine")
        st.write("alterneur : joue C, puis T, puis C, puis T etc...")
        st.write("sympa : joue C toujours, mais joue T à un tour si on l'a trahi au 2 tour précédent")
        st.write("aleatoire_coop : joue C 75% du temps, T 25%")
        st.write("copieur_pardonneur : copie ce qu'a fait son adversaire au tour précedenat, mais pardonne 1/5 des fois")
        st.write("traitre_repenti : joue T 4 fois sur 5, et C 1 fois sur 5")

        st.subheader("Dilemme du prisonnier répété — tournoi")

        # --- Paramètres ---
        nb_partie = st.sidebar.slider("Nombre de tours", 10, 500, 100)
        seed = st.sidebar.number_input("Seed (reproductible)", value=123, step=1)

        # Matrice de gain (standard)
        # (action_j1, action_j2) -> (gain_j1, gain_j2)
        payoff = {
            ("C", "C"): (3, 3),
            ("C", "T"): (0, 5),
            ("T", "C"): (5, 0),
            ("T", "T"): (1, 1),
        }

        # --- Stratégies ---
        def strat_alterneur(my_hist, opp_hist):
            # C, T, C, T, ...
            return "C" if len(my_hist) % 2 == 0 else "T"
        
        def strat_cooperatif(my_hist, opp_hist):
            return "C"

        def strat_traitre(my_hist, opp_hist):
            return "T"

        def strat_copieur(my_hist, opp_hist):
            if len(opp_hist) == 0:
                return "C"
            return opp_hist[-1]
        
        def strat_copieur_pardonneur(my_hist, opp_hist):
            if len(opp_hist) == 0:
                return "C"
            if opp_hist[-1] == "T":
                return "C" if r.random() < 0.2 else "T"
            return "C"

        def strat_rancunier(my_hist, opp_hist):
            # Coopère jusqu'à la première trahison adverse, puis trahit toujours
            if "T" in opp_hist:
                return "T"
            return "C"

        def strat_sympa(my_hist, opp_hist):
            if len(opp_hist) >= 2 and opp_hist[-1] == "T" and opp_hist[-2] == "T":
                return "T"
            return "C"
        
        def strat_vicieux(my_hist, opp_hist):
            if len(opp_hist) >= 3 and opp_hist[-1] == "C" and opp_hist[-2] == "C" and opp_hist[-3] == "C":
                return "T"
            return "C"

        def strat_aleatoire(my_hist, opp_hist):
            return r.choice(["C", "T"])

        def strat_aleatoire_coop(my_hist, opp_hist):
            # 75% C, 25% T
            return "C" if r.random() < 0.75 else "T"
        
        def strat_traitre_repenti(my_hist, opp_hist):
            return "C" if r.random() < 0.2 else "T"

        strategies = {
            "cooperatif": strat_cooperatif,
            "copieur": strat_copieur,
            "copieur_pardonneur": strat_copieur_pardonneur,
            "rancunier": strat_rancunier,
            "aleatoire": strat_aleatoire,
            "aleatoire_coop": strat_aleatoire_coop,
            "alterneur": strat_alterneur,
            "sympa": strat_sympa,
            "traitre": strat_traitre,
            "traitre_repenti": strat_traitre_repenti,
            "vicieux": strat_vicieux
        }

        # --- Moteur de jeu ---
        def jouer_match(f1, f2, n):
            h1, h2 = [], []
            s1 = s2 = 0
            for _ in range(n):
                a1 = f1(h1, h2)
                a2 = f2(h2, h1)
                g1, g2 = payoff[(a1, a2)]
                s1 += g1
                s2 += g2
                h1.append(a1)
                h2.append(a2)
            return s1, s2

        def tournoi(strategies_dict, n):
            noms = list(strategies_dict.keys())
            total_points = {name: 0 for name in noms}
            confrontations = {name: 0.0 for name in noms}

            for i in range(len(noms)):
                for j in range(len(noms)):
                    j1 = noms[i]
                    j2 = noms[j]
                    s1, s2 = jouer_match(strategies_dict[j1], strategies_dict[j2], n)
                    total_points[j1] += s1

                    # "victoire de confrontation" (comme dans ton ancien code)
                    if s1 > s2:
                        confrontations[j1] += 1
                    elif s2 > s1:
                        confrontations[j2] += 1
                    else:
                        confrontations[j1] += 0.5
                        confrontations[j2] += 0.5

            return total_points, confrontations

        # --- UI ---
        st.write("Stratégies incluses :", ", ".join(strategies.keys()))

        if st.button("Lancer la simulation"):
            r.seed(int(seed))
            np.random.seed(int(seed))

            points, victoires = tournoi(strategies, nb_partie)

            # Classements
            classement_points = sorted(points.items(), key=lambda x: x[1], reverse=True)
            classement_victoires = sorted(victoires.items(), key=lambda x: x[1], reverse=True)

            st.markdown("### Classement — points totaux")
            st.table([{"Rang": k+1, "Joueur": name, "Points": score}
                      for k, (name, score) in enumerate(classement_points)])

            st.markdown("### Classement — confrontations gagnées")
            st.table([{"Rang": k+1, "Joueur": name, "Victoires": score}
                      for k, (name, score) in enumerate(classement_victoires)])

            # (optionnel) un exemple de match détaillé
            st.markdown("### Exemple de match (copieur vs traitre)")
            s1, s2 = jouer_match(strategies["copieur"], strategies["traitre"], nb_partie)
            st.write({"copieur": s1, "traitre": s2})    
    # Page 3 (exemple)
    elif page == "Schelling":
 # Paramètres modifiables dans la barre latérale
        size = st.sidebar.slider("Taille de la grille", min_value=10, max_value=100, value=50)
        density = st.sidebar.slider("Densité des agents", min_value=0.0, max_value=1.0, value=0.5)
        type1_ratio = st.sidebar.slider("Proportion de Type 1", min_value=0.0, max_value=1.0, value=0.5)
        tolerance = st.sidebar.slider("Tolérance de satisfaction", min_value=0.0, max_value=1.0, value=0.3)

        # Initialisation de la grille
        grid = init_grid(size, density, type1_ratio)

        # Boucle d'animation
        num_iterations = st.sidebar.slider("Nombre d'itérations", min_value=1, max_value=100, value=10)

        for _ in range(num_iterations):
            grid = update(grid, tolerance)

        # Affichage de la grille
        fig, ax = plt.subplots()
        ax.imshow(grid, cmap='coolwarm', interpolation='nearest')
        ax.set_title("Grille de Ségrégation")
        ax.set_xticks([])
        ax.set_yticks([])
        
        st.pyplot(fig)
# Lancer l'application
if __name__ == "__main__":
    main()
