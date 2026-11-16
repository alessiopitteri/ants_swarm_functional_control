import numpy as np
import networkx as nx   
import matplotlib.pyplot as plt

beta_target = [9.5, 8.5, 7.5, 6.5, 5.5, 4.5, 3.5]
delta_target = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

def create_graph_from_grid(grid):
    rows, cols = grid.shape
    G = nx.Graph()

    # Get all active nodes (where grid == 1)
    nodes = np.argwhere(grid == 1)
    G.add_nodes_from(map(tuple, nodes))  # Convert to tuple for hashability

    # Define the 8 possible moves
    directions = np.array([[-1, -1], [-1, 0], [-1, 1],
                           [0, -1],          [0, 1],
                           [1, -1],  [1, 0],  [1, 1]])

    # Compute valid neighbor coordinates with periodic boundaries
    for r, c in nodes:
        for dr, dc in directions:
            nr = (r + dr) % rows  # Wrap around row boundaries
            nc = (c + dc) % cols  # Wrap around column boundaries
            if grid[nr, nc] == 1:  # Only add edges if the neighbor is also 1
                G.add_edge((r, c), (nr, nc))

    return G
    
def get_largest_connected_component(G):
    try:
        largest_cc = max(nx.connected_components(G), key=len)
        return G.subgraph(largest_cc).copy()
    except ValueError: 
        return nx.Graph() 


def print_latex_block(matrix_m, matrix_M, beta_list, delta_list, bold_threshold=0.80):
    """
    Genera le righe LaTeX per un blocco della tabella (es. baseline, enhanced o smart).
    - matrix_m, matrix_M: matrici 2D numpy contenenti i dati per m(T) e M(T)
    - beta_list, delta_list: i vettori usati per generare la heatmap
    """
    beta_arr = np.array(beta_list)
    delta_arr = np.array(delta_list)

    for b in beta_target:
        b_idx = np.abs(beta_arr - b).argmin()
        
        # Formattazione per m(T)
        cells_m = []
        for d in delta_target:
            d_idx = np.abs(delta_arr - d).argmin()
            val = matrix_m[b_idx, d_idx]
            formatted = f"{val:.2f}"
            cells_m.append(formatted)
            
        # Formattazione per M(T)
        cells_M = []
        for d in delta_target:
            d_idx = np.abs(delta_arr - d).argmin()
            val = matrix_M[b_idx, d_idx]
            formatted = f"{val:.2f}"
            cells_M.append(formatted)
            
        # Stampa riga LaTeX pronta al copia-incolla
        print(f"{b:.1f} & " + " & ".join(cells_m) + " & " + " & ".join(cells_M) + r" \\")

def compute_density(positions, Lx, Ly, n):
    indices = (positions[:n, 0] * Ly + positions[:n, 1]).astype(int)
    rho = np.bincount(indices, minlength=Lx * Ly)
    rho = rho.reshape(Lx, Ly)
    return rho

def order_parameter_s(pheromones, V):
    s0 = np.mean(pheromones)
    plus_id = pheromones > s0
    minus_id = ~plus_id
    
    n_plus = np.sum(plus_id)
    n_minus = np.sum(minus_id)
    
    if n_plus == 0 or n_minus == 0:
        return 0.0
    
    s_plus_avg = np.mean(pheromones[plus_id])
    s_minus_avg = np.mean(pheromones[minus_id])
    
    return ((s_plus_avg - s_minus_avg) / (s0 * V)) * n_plus


def order_parameter_p(pheromones, rho):
    s0 = np.mean(pheromones)
    plus_id = pheromones > s0
    minus_id = ~plus_id
    
    n_plus = np.sum(plus_id)
    n_minus = np.sum(minus_id)
    
    if n_plus == 0 or n_minus == 0:
        return 0.0
    
    mu_plus = np.mean(rho[plus_id])
    mu_minus = np.mean(rho[minus_id])
    
    denom = mu_plus + mu_minus
    if denom == 0:
        return 0.0
        
    return (mu_plus - mu_minus) / denom

def create_graph_from_grid_local(grid):
        rows, cols = grid.shape
        G = nx.Graph()

        # Get all active nodes (grid == 1)
        nodes = np.argwhere(grid == 1)
        node_set = set(map(tuple, nodes))
        G.add_nodes_from(node_set)

        # Define the 8-connected neighborhood
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),          (0, 1),
                      (1, -1),  (1, 0), (1, 1)]

        for r, c in node_set:
            for dr, dc in directions:
                nr, nc = (r + dr), (c + dc)
                if nr < rows and nc < cols and nr > 0 and nc > 0:
                    if grid[nr, nc] == 1 :
                        neighbor = (nr, nc)
                        if neighbor in node_set:
                            G.add_edge((r, c), neighbor)

        return G

def component_map_nodes_degree(env, G):
    component_map = {}

    # initialize the component map with default values for the whole lattice of dimensions Lx and Ly
    component_map = {(x, y): [0, 0] for x in range(env.LL) for y in range(env.Ly)}

    for component in nx.connected_components(G):
        component_nodes = len(component)
        size = component_nodes
        if component_nodes < 2:
            continue

        for node in component:
            degree = G.degree(node)
            component_map[node] = [size, degree]

    return component_map

def global_reward(env):
    
    ns_pher = env.pheromones.copy()
    ns0 = np.mean(ns_pher)
    plus_id = ns_pher > ns0
    grid = plus_id.copy()

    G_t1 = create_graph_from_grid_local(grid)
    components = list(nx.connected_components(G_t1))

    metrics_for_components = []        

    for component in components:
        n_nodes = len(component)
        if n_nodes < 2:
            continue
        n_edges = G_t1.subgraph(component).number_of_edges()
        edge_pen = -n_edges/(n_nodes-1)

        pop = env.order_parameter_s()
        pod = env.order_parameter_p()

        len_rew = min(n_nodes,env.Lx)/env.Lx
        rew = edge_pen + (pop*len_rew+pod)/2
    
        metrics_for_components.append([n_edges, n_nodes, rew])

    return np.array(metrics_for_components)


def plot_heatmap(metric, delta_list, beta_list, cmap='summer', title=None, vmin=None, vmax=None, beta_crit_values=None):
    """
    Genera un heatmap con estensione centrata sui pixel, tick allineati
    e etichette in formato LaTeX.
    """
    # Calcolo degli step per centrare i pixel attorno ai punti
    d_delta = (max(delta_list) - min(delta_list)) / (len(delta_list) - 1) if len(delta_list) > 1 else 0.5
    d_beta  = (max(beta_list)  - min(beta_list))  / (len(beta_list)  - 1) if len(beta_list)  > 1 else 0.5

    extent = [
        min(delta_list) - d_delta / 2,
        max(delta_list) + d_delta / 2,
        min(beta_list)  - d_beta  / 2,
        max(beta_list)  + d_beta  / 2
    ]

    plt.figure()
    plt.imshow(metric, cmap=cmap, extent=extent, origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar()

    # Tick al centro dei pixel
    plt.xticks(delta_list)
    plt.yticks(beta_list)

    # Etichette in LaTeX (usa r'$\Delta$' se preferisci la lettera maiuscola)
    plt.xlabel(r'$\delta$')  
    plt.ylabel(r'$\beta$')

    if title:
        plt.title(title)

    if beta_crit_values is not None:
        plt.plot(delta_list, beta_crit_values, color='red', linestyle='--', label=r'$\beta_{crit}$')
    plt.tight_layout()
    plt.show()


    # =====================================================================
# FUNZIONI HELPER INTERNE (evitano duplicazione di codice tra le metriche)
# =====================================================================
def _get_visit_probabilities(pos_window: np.ndarray) -> np.ndarray:
    """Restituisce la matrice p_ik di probabilità di visita (num_cells, num_ants)."""
    num_timesteps, num_ants, _ = pos_window.shape
    flat_pos = pos_window.reshape(-1, 2)
    _, inverse_indices = np.unique(flat_pos, axis=0, return_inverse=True)
    cell_ids = inverse_indices.reshape(num_timesteps, num_ants)
    num_cells = np.max(cell_ids) + 1

    return np.column_stack([
        np.bincount(cell_ids[:, k], minlength=num_cells) / num_timesteps
        for k in range(num_ants)
    ])


def _shannon_entropy(probs: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """Calcola l'entropia di Shannon in base 2 gestendo gli zeri."""
    with np.errstate(divide="ignore", invalid="ignore"):
        log2_p = np.where(probs > 0, np.log2(probs), 0.0)
        return -np.sum(probs * log2_p, axis=axis)


def _compute_pbc_displacements(pos_window: np.ndarray, L: float) -> np.ndarray:
    """Calcola gli spostamenti passo-passo corretti per le condizioni periodiche al contorno."""
    dx = np.diff(pos_window, axis=0)
    return dx - L * np.round(dx / L)


# =====================================================================
# FUNZIONI DEDICATE PER CIASCUNA METRICA
# =====================================================================
def compute_collective_entropy(pos_window: np.ndarray, L: float) -> float:
    """Calcola l'entropia collettiva normalizzata rispetto a log2(L^2)."""
    p_ik = _get_visit_probabilities(pos_window)
    p_i = np.mean(p_ik, axis=1)
    h_coll = _shannon_entropy(p_i)
    return float(h_coll / np.log2(L**2))


def compute_individual_entropy(pos_window: np.ndarray) -> tuple[float, float]:
    """Calcola media e deviazione standard dell'entropia individuale."""
    p_ik = _get_visit_probabilities(pos_window)
    entropies = _shannon_entropy(p_ik, axis=0)
    return float(np.mean(entropies)), float(np.std(entropies))


def compute_mutual_information(pos_window: np.ndarray) -> float:
    """Calcola la mutual information I = H_coll - <H_ind> (non normalizzata)."""
    p_ik = _get_visit_probabilities(pos_window)
    p_i = np.mean(p_ik, axis=1)
    h_coll = _shannon_entropy(p_i)
    h_ind = np.mean(_shannon_entropy(p_ik, axis=0))
    return float(h_coll - h_ind)


def compute_spatial_extent(pos_window: np.ndarray) -> float:
    """Calcola l'estensione spaziale media E (numero medio di celle uniche visitate)."""
    p_ik = _get_visit_probabilities(pos_window)
    return float(np.mean(np.sum(p_ik > 0, axis=0)))


def compute_msd(pos_window: np.ndarray, L: float, lag_list: list[int]) -> np.ndarray:
    """Calcola lo spostamento medio (con unwrapping PBC) per ciascun lag."""
    tau = pos_window.shape[0]
    dx_pbc = _compute_pbc_displacements(pos_window, L)

    pos_unwrap = np.empty_like(pos_window)
    pos_unwrap[0] = pos_window[0]
    pos_unwrap[1:] = pos_window[0] + np.cumsum(dx_pbc, axis=0)

    msd_l = np.full(len(lag_list), np.nan)
    for idx, lag in enumerate(lag_list):
        if lag < tau:
            disp = pos_unwrap[lag:] - pos_unwrap[:-lag]
            msd_l[idx] = np.mean(np.sqrt(np.sum(disp**2, axis=-1)))

    return msd_l


def compute_directional_persistence(
    pos_window: np.ndarray, L: float, lag_list: list[int]
) -> np.ndarray:
    """Calcola la persistenza direzionale C(l) per ciascun lag."""
    dx_pbc = _compute_pbc_displacements(pos_window, L)
    step_norms = np.linalg.norm(dx_pbc, axis=-1, keepdims=True)
    u_k = np.where(step_norms > 1e-8, dx_pbc / step_norms, 0.0)

    c_l = np.full(len(lag_list), np.nan)
    for idx, lag in enumerate(lag_list):
        if lag < len(u_k):
            dot_products = np.sum(u_k[:-lag] * u_k[lag:], axis=-1)
            c_l[idx] = np.mean(dot_products)

    return c_l