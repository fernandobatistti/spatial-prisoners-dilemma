import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import time
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from scipy.ndimage import label, convolve
from numba import njit, prange

# ==============================================================================
# 1. CONTROLE DE EXECUÇÃO E PARÂMETROS GERAIS
# ==============================================================================

GERAR_FRAMES = True            # True = Cria imagens (lento) | False = Apenas CSV (muito rápido)
CALCULAR_CINEMATICA = True    # True = Extrai Rg2, MSD e Anisotropia. False = Desativa cálculos pesados e oculta marcadores visuais.
# Se for um número inteiro (ex: 50), o código gera 50 sementes aleatórias únicas e as salva em um .txt.
# Se for uma lista (ex: [42, 105, 999]), o código roda a simulação para essas sementes exatas.
CONFIG_SEMENTES = [44] #[7412853, 9158204, 3429187, 8204619, 5138472, 6391054, 1847293, 4920183, 7654329, 2384915, 8719245, 1294857, 5638201, 9482013, 3720194, 6503928, 4182937, 7839201, 2049183, 8391024, 5928104, 1473928, 6839204, 9204815, 3184920, 7549182, 4291083, 8603912, 1958204, 6720491, 3849102, 8192047, 5403918, 9720481, 2618493, 7384910, 4859201, 6204918, 1538294, 8947201, 3294810, 7183920, 5849201, 9038471, 2471938, 6958204, 4381920, 8749102, 1629483, 7938401, 5284910, 9384710, 3619482, 8049182, 2859301, 6472910, 4719283, 8520391, 1394820, 7284910, 5739104, 9603819, 3948201, 8472910, 2194830, 6849102, 4529183, 7739102, 1829470, 8294018, 5048192, 9182736, 3571928, 8649201, 2938471, 6384912, 4193820, 7492018, 1748293, 8839201, 5392018, 9503819, 3284719, 8103928, 2491830, 6739104, 4958201, 7820491, 1263849, 8402918, 5829104, 9748201, 3819204, 8592014, 2738491, 6194820, 4403918, 7619482, 1948203, 8920381, 5194820, 9304819, 3472918, 8739102, 2084913, 6592018, 4829104, 7304918, 1482930, 8173920, 5648201, 9820391, 3749201, 8364910, 2583910, 6918204, 4271938, 7948201, 1673928, 8519482, 5482910, 9093847, 3194820, 8820491, 2839401, 6284719, 4619382, 7503918, 1184920, 8763910, 5948201, 9438291, 3629104, 8249180, 2374918, 6794820, 4082937, 7859201, 1594820, 8319482, 5273910, 9684910, 3904819, 8938471, 2748192, 6439102, 4758201, 7193840, 1374928, 8684912, 5794820, 9248193, 3384910, 8492018, 2184903, 6629481, 4392810, 7704918, 1863924, 8094821, 5539201, 9794820, 3859201, 8629401, 2948301, 6148293, 4918204, 7438291, 1074928, 8958204, 5381940, 9148293, 3504918, 8382910, 2483910, 6874920, 4539182, 7984201, 1729483, 8264910, 5864910, 9593820, 3248190, 8704918, 2693841, 6319482, 4173928, 7649182, 1295840, 8894720, 5128493, 9884920, 3782910, 8439102, 2264918, 6549201, 4892014, 7239481, 1459283, 8564912, 2988373, 1631369, 3094653, 692374, 786320, 2743236, 2294249, 8477433, 4579164, 8533878, 1148344, 6979476, 7993927, 2943381, 406517, 8638813, 6752332, 2776420, 4674664, 6605089, 2980638, 6143697, 8025147, 7509466, 5198571, 6171560, 8843643, 1122338, 7881451, 5961642, 1000115, 6487195, 3693178, 1850517, 7365776, 9313348, 7778543, 7269986, 7829351, 5787101, 2106607, 4320663, 4217723, 4499374, 4624898, 1245694, 5257297, 1092224, 2519198, 9592857, 9857883, 6749050, 4058198, 1948765, 1312746, 9630288, 8034224, 7866400, 4405177, 5215132, 3133369, 1872491, 1896274, 3118375, 2385546, 3861856, 84582, 4582271, 5003947, 615467, 7154766, 7617923, 6872165, 1273682, 2157774, 4457601, 6010906, 742715, 2907082, 487322, 6616805, 6498229, 605123, 812624, 4806513, 3518850, 4672606, 4625049, 3113407, 3643621, 1740938, 7665415, 2365715, 2067302, 8992256, 8686818, 5592438, 5777822, 5017746, 1343189]

FRAME_SKIP = 1000                  # Se GERAR_FRAMES for True, a cada quantos passos salvar uma imagem?

# ----- SISTEMA DE SUPERACOMPUTAÇÃO (HPC) -----
CONTINUAR_SIMULACAO = True    # True = Busca checkpoints para retomar de onde parou. False = Inicia do zero e sobrescreve.
CHECKPOINT_SKIP = 1000        # A cada quantos passos o código salva o "save state" do jogo?
TAMANHOS_REDE = [201]           # Lista de tamanhos. Ex: [101, 201, 400] para análise de Finite-Size Scaling
STEPS = 20000                     # Número de passos da simulação

# Parâmetros do Dilema do prisioneiro (Payoff Matrix)
b = 1.2  # Tentação (T). 
PAYOFF_MATRIX = np.array([
    [1.0,  0.0],      
    [b,    0.01]
], dtype=np.float64)

TOPOLOGIA_TABULEIRO = "toroide"     # "toroide" ou "parede"
TIPO_VIZINHANCA = "moore"           # "moore" ou "neumann"
PROFUNDIDADE_VIZINHANCA = 1         # Profundidade da vizinhança (1 = 8 vizinhos, 2 = 24 vizinhos, etc.)
PROBABILIDADE_ATUALIZACAO = 1.0     # Se for 1.0, todos os agentes atualizam simultaneamente. Se for < 1.0, cada agente tem uma chance de atualizar a cada passo.

# Resgate Mutacional
TAXA_MUTACAO_C = 0.0001              # Taxa de mutação para a estratégia C (Representa uma probabilidade do indivíduo mudar de estratégia a cada passo)
TAXA_MUTACAO_D = 0.0001            # Taxa de mutação para a estratégia D (Representa uma probabilidade do indivíduo mudar de estratégia a cada passo)

# Condição Inicial (Geografia)
ESTRATEGIA_MAR = 1                                                              # 0 = Cooperador, 1 = Desertor, 2 = Buraco
ESTRATEGIA_INVASOR = 0                                                          # 0 = Cooperador, 1 = Desertor, 2 = Buraco
CONFIG_INVASORES = [{"tipo": "cartesiano", "x": 0, "y": 0, "tamanho": 9}
                    ] # [{"tipo": "cartesiano", "x": 0, "y": 0, "tamanho": 9}] Lista de dicionários representando os invasores. Cada dicionário deve conter as chaves: "tipo" (cartesiano ou polar), "x" e "y" (ou "r" e "theta" se polar), e "tamanho" (tamanho do bloco de invasores).
CONFIG_BURACOS = []                                                             # Lista de dicionários representando os buracos. Cada dicionário deve conter as chaves: "tipo" (cartesiano ou polar), "x" e "y" (ou "r" e "theta" se polar), e "tamanho" (tamanho do bloco de buracos).

# O limiar da percolação é aproximadamente 0.407254 de buracos para vizinhança de Neumann e 0.592746 de buracos para vizinhança de Moore. Fonte: Havlin
PROPORCAO_RANDOM_BURACOS = 0.407254                                     # Proporção de buracos a serem gerados aleatoriamente.

# Se for 0.0, o código usa a semente do CONFIG_INVASORES no centro. 
# Se for > 0.0 (ex: 0.10), o código espalha cooperadores aleatoriamente no espaço viável.
PROPORCAO_RANDOM_COOPERADORES = 0.0                                            # Proporção de cooperadores a serem gerados aleatoriamente.

INDICE_C = 0                # Índice do cooperador definido como 0
INDICE_D = 1                # Índice do desertor definido como 1
INDICE_BURACO = 2           # Índice do buraco definido como 2
NUM_STRATEGIES = 2          # Número de estratégias (C e D). O buraco não é considerado uma estratégia ativa, mas sim um estado do ambiente.

# ==============================================================================
# REGRAS DE APRENDIZAGEM E RUÍDO TÉRMICO (REGRA DE FERMI)
# ==============================================================================
# Se K_FERMI = 0.0, o sistema usa "Imitate the Best" (Racionalidade Máxima - Seu modelo original).
# Se K_FERMI > 0.0, o sistema usa a Regra de Fermi (Abre margem para imitação irracional).
K_FERMI = 0.0          # Temperatura / Irracionalidade (ex: 0.1 no artigo da Nature)
TAU_FERMI = 0.0        # Custo de atrito/mudança (ex: 0.0 ou 0.1)    

# ==============================================================================
# 1.5 GERAÇÃO DINÂMICA DO NOME DO EXPERIMENTO E DIRETÓRIOS
# ==============================================================================
# O f-string extrai as variáveis e as formata com casas decimais precisas.
NOME_EXPERIMENTO = f"b{b:.2f}_mutC{TAXA_MUTACAO_C:.4f}_mutD{TAXA_MUTACAO_D:.4f}_pbur{PROPORCAO_RANDOM_BURACOS:.3f}_pcoop{PROPORCAO_RANDOM_COOPERADORES:.3f}_viz{TIPO_VIZINHANCA}_upd{PROBABILIDADE_ATUALIZACAO:.2f}_prof{PROFUNDIDADE_VIZINHANCA}_{TOPOLOGIA_TABULEIRO}"

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
CAMINHO_DADOS = os.path.join(BASE_DIR, NOME_EXPERIMENTO, 'data') 
CAMINHO_FRAMES = os.path.join(BASE_DIR, NOME_EXPERIMENTO, 'frames') 
CAMINHO_CHECKPOINTS = os.path.join(BASE_DIR, NOME_EXPERIMENTO, 'checkpoints')

os.makedirs(CAMINHO_DADOS, exist_ok=True) 
os.makedirs(CAMINHO_CHECKPOINTS, exist_ok=True)
if GERAR_FRAMES: 
    os.makedirs(CAMINHO_FRAMES, exist_ok=True) 
    
# ==============================================================================
# CORES PARA RENDERIZAÇÃO
# ==============================================================================

COLORS = np.array([
    [0.12, 0.46, 0.70, 1.0],  # Azul (Cooperadores)
    [0.83, 0.15, 0.15, 1.0],  # Vermelho (Desertores)
    [0.00, 0.00, 0.00, 1.0]   # Preto (Buracos)
])

# ==============================================================================
# 2. GERADORES DE VIZINHANÇA E GEOMETRIA
# ==============================================================================
def gerar_vizinhanca(tipo, profundidade):
    tamanho = 2 * profundidade + 1                                                              # O tamanho do kernel é determinado pela profundidade da vizinhança. Por exemplo, uma profundidade de 1 resulta em um kernel 3x3, enquanto uma profundidade de 2 resulta em um kernel 5x5.
    centro = profundidade                                                                       # O centro do kernel é a posição do agente central. Por exemplo, em um kernel 3x3, o centro está na posição (1,1), enquanto em um kernel 5x5, o centro está na posição (2,2).
    kernel = np.zeros((tamanho, tamanho), dtype=np.int32)                                       # Cria uma matriz de zeros do tamanho do kernel. Esta matriz será preenchida com 1s nas posições que representam os vizinhos do agente central, de acordo com o tipo de vizinhança (Moore ou Neumann) e a profundidade especificada. 
    directions = []
    
    for i in range(tamanho):
        for j in range(tamanho):
            dx = i - centro
            dy = j - centro
            if dx == 0 and dy == 0:
                # Centro do kernel: não adiciona a si mesmo
                continue 
            if tipo == "moore" and max(abs(dx), abs(dy)) <= profundidade:
                kernel[i, j] = 1          
                directions.append((dx, dy)) 
            elif tipo == "neumann" and abs(dx) + abs(dy) <= profundidade:
                kernel[i, j] = 1          
                directions.append((dx, dy)) 
    return kernel, directions

KERNEL_INTERACAO, DIRECTIONS = gerar_vizinhanca(TIPO_VIZINHANCA, PROFUNDIDADE_VIZINHANCA)
DIRECTIONS_ARRAY = np.array(DIRECTIONS, dtype=np.int32)

def coordenadas_para_matriz(config, L):
    centro_x_continuo = (L - 1) / 2.0
    centro_y_continuo = (L - 1) / 2.0
    if config["tipo"] == "cartesiano":
        x, y = config["x"], config["y"]
    elif config["tipo"] == "polar":
        r, theta_rad = config["r"], np.radians(config["theta"])
        x, y = r * np.cos(theta_rad), r * np.sin(theta_rad)
    tamanho = config["tamanho"]
    start_c = int(round(centro_x_continuo + x - (tamanho - 1) / 2.0))
    start_r = int(round(centro_y_continuo - y - (tamanho - 1) / 2.0))
    return start_r, start_c

def desenhar_bloco(grid, start_r, start_c, tamanho, valor, L):
    for r in range(start_r, start_r + tamanho):
        for c in range(start_c, start_c + tamanho):
            grid[r % L, c % L] = valor
    return grid

def inicializar_universo(L):
    
    # 1. Nasce todo mundo Desertor por padrão
    grid = np.full((L, L), ESTRATEGIA_MAR, dtype=np.int32)
    hole_mask = np.zeros((L, L), dtype=bool)
    paredes_mask = np.zeros((L, L), dtype=bool) 
    
    # 2. Injeta Invasores Manuais (se a proporção randômica estiver desligada)
    if PROPORCAO_RANDOM_COOPERADORES == 0.0:
        for inv in CONFIG_INVASORES:
            r, c = coordenadas_para_matriz(inv, L)
            grid = desenhar_bloco(grid, r, c, inv["tamanho"], ESTRATEGIA_INVASOR, L)
            
    # 3. Injeta Buracos Manuais
    for buraco in CONFIG_BURACOS:
        r, c = coordenadas_para_matriz(buraco, L)
        temp_grid = np.zeros_like(grid, dtype=bool)
        temp_grid = desenhar_bloco(temp_grid, r, c, buraco["tamanho"], True, L)
        hole_mask |= temp_grid
        
    # ======================================================================
    # 4. Injeta a Desordem Topológica (DETERMINISMO PROPORCIONAL)
    # ======================================================================
    if PROPORCAO_RANDOM_BURACOS > 0:
        # Pega as coordenadas apenas de quem NÃO é buraco manual ainda
        celulas_livres = np.where(~hole_mask)
        num_livres = len(celulas_livres[0])
        
        # Calcula a quantidade exata cravada com base em 100% do L*L
        qtd_buracos = int(round((L * L) * PROPORCAO_RANDOM_BURACOS))
        
        # Trava de segurança: não permite tentar criar mais buracos que o espaço livre
        qtd_buracos = min(qtd_buracos, num_livres)
        
        # Sorteia exatamente as posições únicas (replace=False)
        indices_sorteados = np.random.choice(num_livres, qtd_buracos, replace=False)
        
        # Mapeia os índices sorteados de volta para as coordenadas reais da matriz
        coords_r = celulas_livres[0][indices_sorteados]
        coords_c = celulas_livres[1][indices_sorteados]
        hole_mask[coords_r, coords_c] = True
        
    grid[hole_mask] = INDICE_BURACO
    
    # ======================================================================
    # 5. POVOAMENTO ALEATÓRIO (DETERMINISMO NO "Novo 100%")
    # ======================================================================
    if PROPORCAO_RANDOM_COOPERADORES > 0.0:
        # O espaço vital são as células que sobraram após injetar buracos
        celulas_validas = np.where(~hole_mask)
        num_validas = len(celulas_validas[0])
        
        # Calcula a cota exata cravada de cooperadores
        qtd_coops = int(round(num_validas * PROPORCAO_RANDOM_COOPERADORES))
        
        # Sorteia as posições exatas únicas
        indices_sorteados = np.random.choice(num_validas, qtd_coops, replace=False)
        
        # Carimba os Cooperadores
        coords_r = celulas_validas[0][indices_sorteados]
        coords_c = celulas_validas[1][indices_sorteados]
        grid[coords_r, coords_c] = ESTRATEGIA_INVASOR
    # ======================================================================
        
    # 6. Paredes Intransponíveis (Proteção Absoluta das Bordas)
    if TOPOLOGIA_TABULEIRO == "parede":
        d = PROFUNDIDADE_VIZINHANCA
        paredes_mask[:d, :] = True
        paredes_mask[-d:, :] = True
        paredes_mask[:, :d] = True
        paredes_mask[:, -d:] = True
        hole_mask |= paredes_mask
        grid[paredes_mask] = INDICE_BURACO
        
    return grid, paredes_mask

# ==============================================================================
# 3. FUNÇÕES FÍSICAS E EXCELÊNCIA CINEMÁTICA
# ==============================================================================

# O @njit(parallel=True) avisa o Numba para transformar isso em C e usar Multi-Core
@njit(parallel=True)
def censo_e_payoff_numba(grid, L, payoff_c_c, payoff_c_d, payoff_d_c, payoff_d_d, directions):
    payoffs = np.zeros((L, L), dtype=np.float64)
    n_c_matrix = np.zeros((L, L), dtype=np.int32)
    n_d_matrix = np.zeros((L, L), dtype=np.int32)
    
    num_vizinhos = directions.shape[0] # Quantos vizinhos existem na topologia atual?
    
    for r in prange(L):
        for c in range(L):
            n_c = 0
            n_d = 0
            
            # Varre os vizinhos de forma dinâmica baseada na topologia
            for i in range(num_vizinhos):
                dr = directions[i, 0]
                dc = directions[i, 1]
                
                # Efeito Toroide (%)
                nr = (r + dr) % L
                nc = (c + dc) % L
                
                viz_strat = grid[nr, nc]
                if viz_strat == 0:   # Cooperador
                    n_c += 1
                elif viz_strat == 1: # Desertor
                    n_d += 1
                        
            n_c_matrix[r, c] = n_c
            n_d_matrix[r, c] = n_d
            
            # Calcula Payoff instantaneamente
            strat = grid[r, c]
            if strat == 0:
                payoffs[r, c] = n_c * payoff_c_c + n_d * payoff_c_d
            elif strat == 1:
                payoffs[r, c] = n_c * payoff_d_c + n_d * payoff_d_d
                
    return n_c_matrix, n_d_matrix, payoffs

@njit(parallel=True)
def escolha_racional_numba(grid, payoffs, L, K_FERMI, TAU_FERMI, directions):

    # Cria a grade do futuro
    proposed_grid = np.copy(grid)
    num_vizinhos = directions.shape[0]
    
    for r in prange(L):
        for c in range(L):
            # Buracos não tomam decisões
            if grid[r, c] == 2: 
                continue
            
            # ==================================================================
            # MUNDO 1: O SEU MOTOR ORIGINAL (Temperatura Zero - Determinístico)
            # ==================================================================
            if K_FERMI <= 0.0:
                best_payoff = payoffs[r, c]  # A melhor nota começa sendo a do PRÓPRIO jogador
                minha_estrategia = grid[r, c]
                best_strat = minha_estrategia      # A estratégia vencedora começa sendo a DELE MESMO

                # Varre os vizinhos de forma dinâmica baseada na topologia
                for i in range(num_vizinhos):
                    dr = directions[i, 0]
                    dc = directions[i, 1]

                    # Efeito Toroide (%)
                    nr = (r + dr) % L
                    nc = (c + dc) % L

                    # Pega a estratégia e o payoff do vizinho
                    viz_strat = grid[nr, nc]
                    viz_payoff = payoffs[nr, nc]

                    # Só considera vizinhos válidos (não buracos)
                    if viz_strat != 2: 
                        if viz_payoff > best_payoff:
                            # Se o vizinho tem nota estritamente maior, ele assume a liderança
                            best_payoff = viz_payoff
                            best_strat = viz_strat
                        elif viz_payoff == best_payoff:
                            # CRITÉRIO DE DESEMPATE BIOLÓGICO (INÉRCIA):
                            # Se houver empate na nota máxima e um dos líderes tiver a mesma 
                            # estratégia que a célula focal, o conservadorismo dita manter a estratégia.
                            if viz_strat == minha_estrategia:
                                best_strat = minha_estrategia
                                
                proposed_grid[r, c] = best_strat
                
            # ==================================================================
            # MUNDO 2: (Regra de Fermi - Estocástico)
            # ==================================================================
            else:
                # Sorteia aleatoriamente UM índice do array de direções
                idx_sorteado = np.random.randint(0, num_vizinhos)

                # Pega o vizinho sorteado
                dr = directions[idx_sorteado, 0]
                dc = directions[idx_sorteado, 1]

                # Efeito Toroide (%)
                nr = (r + dr) % L
                nc = (c + dc) % L
                viz_strat = grid[nr, nc]

                # Só considera vizinhos válidos (não buracos)
                if viz_strat != 2:
                    pi_x = payoffs[r, c]     
                    pi_y = payoffs[nr, nc]   

                    # Calcula a probabilidade de troca usando a Regra de Fermi
                    expoente = (pi_x - pi_y + TAU_FERMI) / K_FERMI
                    if expoente > 50.0:
                        prob_troca = 0.0
                    elif expoente < -50.0:
                        prob_troca = 1.0
                    else:
                        prob_troca = 1.0 / (1.0 + np.exp(expoente))
                    
                    # A Roleta de Probabilidade: Mudo de estratégia ou continuo?
                    if np.random.rand() < prob_troca:
                        proposed_grid[r, c] = viz_strat
                        
    return proposed_grid

@njit(cache=True)
def label_torus_numba_buf(is_c, L, labels, queue_r, queue_c, offset_x, offset_y, percola_x, percola_y):
    labels[:, :] = 0
    offset_x[:, :] = 0
    offset_y[:, :] = 0
    percola_x[:] = False
    percola_y[:] = False
    current_label = 0
    
    for r in range(L):
        for c in range(L):
            if is_c[r, c] and labels[r, c] == 0:
                current_label += 1
                queue_r[0] = r; queue_c[0] = c
                labels[r, c] = current_label
                head = 0; tail = 1
                
                while head < tail:
                    curr_r = queue_r[head]; curr_c = queue_c[head]; head += 1
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0: continue
                            nr = (curr_r + dr) % L
                            nc = (curr_c + dc) % L
                            
                            # Calcula o deslocamento acumulado
                            dox = offset_x[curr_r, curr_c] + (1 if curr_c + dc >= L else (-1 if curr_c + dc < 0 else 0))
                            doy = offset_y[curr_r, curr_c] + (1 if curr_r + dr >= L else (-1 if curr_r + dr < 0 else 0))
                            
                            if is_c[nr, nc]:
                                if labels[nr, nc] == 0:
                                    labels[nr, nc] = current_label
                                    offset_x[nr, nc] = dox
                                    offset_y[nr, nc] = doy
                                    queue_r[tail] = nr; queue_c[tail] = nc
                                    tail += 1
                                elif labels[nr, nc] == current_label:
                                    # Se encontrou o mesmo cluster com offset diferente, PERCOLOU!
                                    if offset_x[nr, nc] != dox: percola_x[current_label] = True
                                    if offset_y[nr, nc] != doy: percola_y[current_label] = True
    return current_label

def calcular_cinematica_completa(is_c, L, topologia, labels_buf, queue_r_buf, queue_c_buf, offset_x_buf, offset_y_buf, percola_x_buf, percola_y_buf, cx_prev=np.nan, cy_prev=np.nan):
    num_clusters = label_torus_numba_buf(is_c, L, labels_buf, queue_r_buf, queue_c_buf, offset_x_buf, offset_y_buf, percola_x_buf, percola_y_buf)
    matriz_rotulada = labels_buf
    
    if num_clusters == 0: return np.nan, np.nan, np.nan, np.nan, 0, False, False
        
    tamanhos = np.bincount(matriz_rotulada.ravel())
    tamanhos[0] = 0 
    
    rotulos_validos = np.where(tamanhos >= 4)[0]
    if len(rotulos_validos) == 0: return np.nan, np.nan, np.nan, np.nan, 0, False, False
        
    melhor_rotulo = rotulos_validos[np.argmax(tamanhos[rotulos_validos])]
    
    # Flags exatas de percolação topológica para o cluster dominante
    p_x = percola_x_buf[melhor_rotulo]
    p_y = percola_y_buf[melhor_rotulo]
    
    ys, xs = np.where(matriz_rotulada > 0)
    rotulos_pt = matriz_rotulada[ys, xs]
    ordem = np.argsort(rotulos_pt)
    ys, xs, rotulos_pt = ys[ordem], xs[ordem], rotulos_pt[ordem]
    limites = np.searchsorted(rotulos_pt, rotulos_validos)

    if not np.isnan(cx_prev) and not np.isnan(cy_prev):
        melhor_score = np.inf
        for k, rotulo in enumerate(rotulos_validos):
            ini = limites[k]; fim = ini + tamanhos[rotulo]
            pos_y, pos_x = ys[ini:fim], xs[ini:fim]
            massa_cluster = tamanhos[rotulo]
            
            if topologia == "toroide":
                theta_x_cand = 2 * np.pi * pos_x / L
                theta_y_cand = 2 * np.pi * pos_y / L
                cm_x_cand = (np.arctan2(np.mean(np.sin(theta_x_cand)), np.mean(np.cos(theta_x_cand))) * L / (2 * np.pi)) % L
                cm_y_cand = (np.arctan2(np.mean(np.sin(theta_y_cand)), np.mean(np.cos(theta_y_cand))) * L / (2 * np.pi)) % L
                dx_min = np.minimum(np.abs(cm_x_cand - cx_prev), L - np.abs(cm_x_cand - cx_prev))
                dy_min = np.minimum(np.abs(cm_y_cand - cy_prev), L - np.abs(cm_y_cand - cy_prev))
            else:
                dx_min = np.abs(np.mean(pos_x) - cx_prev)
                dy_min = np.abs(np.mean(pos_y) - cy_prev)
                
            dist = dx_min**2 + dy_min**2
            score = (dist + 1.0) / (massa_cluster ** 2)
            if score < melhor_score:
                melhor_score = score
                melhor_rotulo = rotulo

    idx_vencedor = np.where(rotulos_validos == melhor_rotulo)[0][0]
    ini_v = limites[idx_vencedor]; fim_v = ini_v + tamanhos[melhor_rotulo]
    pos_y_matrix = ys[ini_v:fim_v]
    pos_x_matrix = xs[ini_v:fim_v]

    anisotropia = np.nan
    rg2_escalar = np.nan

    if topologia == "toroide":
        theta_x = 2 * np.pi * pos_x_matrix / L
        theta_y = 2 * np.pi * pos_y_matrix / L
        S_x, C_x = np.mean(np.sin(theta_x)), np.mean(np.cos(theta_x))
        S_y, C_y = np.mean(np.sin(theta_y)), np.mean(np.cos(theta_y))
        
        R_x = np.sqrt(S_x**2 + C_x**2)
        R_y = np.sqrt(S_y**2 + C_y**2)
        
        # O CM Circular Contínuo (Excelente para Voos de Lévy)
        cm_x = (np.arctan2(S_x, C_x) * L / (2 * np.pi)) % L
        cm_y = (np.arctan2(S_y, C_y) * L / (2 * np.pi)) % L
        
        # A volta da Imagem Mínima Clássica, que satura em L^2/6 sem divergir
        dx = pos_x_matrix - cm_x
        dy = pos_y_matrix - cm_y
        dx = dx - L * np.round(dx / L) 
        dy = dy - L * np.round(dy / L)
        
        S_xx, S_yy, S_xy = np.mean(dx**2), np.mean(dy**2), np.mean(dx * dy)
        rg2_escalar = S_xx + S_yy
        
        autovalores = np.linalg.eigvalsh(np.array([[S_xx, S_xy], [S_xy, S_yy]]))
        if rg2_escalar > 0: anisotropia = 1.0 - 4.0 * (autovalores[0] * autovalores[1]) / (rg2_escalar**2)
            
    else: # "parede"
        cm_x, cm_y = np.mean(pos_x_matrix), np.mean(pos_y_matrix)
        dx, dy = pos_x_matrix - cm_x, pos_y_matrix - cm_y
        S_xx, S_yy, S_xy = np.mean(dx**2), np.mean(dy**2), np.mean(dx * dy)
        rg2_escalar = S_xx + S_yy
        autovalores = np.linalg.eigvalsh(np.array([[S_xx, S_xy], [S_xy, S_yy]]))
        if rg2_escalar > 0: anisotropia = 1.0 - 4.0 * (autovalores[0] * autovalores[1]) / (rg2_escalar**2)
            
    return cm_x, cm_y, rg2_escalar, anisotropia, tamanhos[melhor_rotulo], p_x, p_y

# ==============================================================================
# 4. RENDERIZAÇÃO DE IMAGENS (OPCIONAL)
# ==============================================================================
def renderizar_frame(grid, t, cx, cy, rg2, anisotropia, msd, L, paredes_mask, semente):
    # Formato Widescreen (16:9)
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    
    # gs[0, 0] é a Esquerda (Tamanho 3) | gs[0, 1] é a Direita (Tamanho 1.2)
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1.2])
    
    ax_board = fig.add_subplot(gs[0, 0])
    ax_text = fig.add_subplot(gs[0, 1])
    
    # ==========================================================
    # 1. RENDERIZAÇÃO DO TABULEIRO (ESQUERDA)
    # ==========================================================
    cmap = mcolors.ListedColormap(COLORS)
    ax_board.imshow(grid, cmap=cmap, vmin=0, vmax=INDICE_BURACO, 
                    aspect='equal', extent=(-0.5, L-0.5, L-0.5, -0.5))
    
    ax_board.set_xticks(np.arange(-0.5, L, 1))
    ax_board.set_yticks(np.arange(-0.5, L, 1))
    ax_board.set_xticklabels([])
    ax_board.set_yticklabels([])
    ax_board.grid(color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    ax_board.set_xlim(-0.5, L - 0.5)
    ax_board.set_ylim(L - 0.5, -0.5)
    
    # Desenha o Centro de Massa
    if not np.isnan(cx):
        ax_board.scatter(cx, cy, color='white', marker='+', s=250, linewidths=1.5)
        ax_board.add_patch(plt.Circle((cx, cy), 1.5, color='white', fill=False, alpha=0.8, lw=2))

    # ==========================================================
    # 2. CÁLCULO DE ESTATÍSTICAS E RENORMALIZAÇÃO
    # ==========================================================
    total_uteis = (L * L) - np.sum(paredes_mask)
    num_c = np.sum(grid == INDICE_C)
    num_d = np.sum(grid == INDICE_D)
    num_p = np.sum((grid == INDICE_BURACO) & (~paredes_mask))
    
    frac_c_global = num_c / total_uteis if total_uteis > 0 else 0.0
    frac_d_global = num_d / total_uteis if total_uteis > 0 else 0.0
    frac_p_global = num_p / total_uteis if total_uteis > 0 else 0.0
    
    # Cálculo da Densidade Populacional Relativa (O nosso Parâmetro de Ordem real)
    pop_ativa = num_c + num_d
    if pop_ativa > 0:
        frac_c_rel = num_c / pop_ativa
        frac_d_rel = num_d / pop_ativa
    else:
        frac_c_rel = frac_d_rel = 0.0

    # ==========================================================
    # 3. PAINEL DE INFORMAÇÕES (DIREITA)
    # ==========================================================
    ax_text.axis('off') 
    fonte_padrao = 'serif'
    
    # Títulos
    ax_text.text(0.0, 0.95, "DILEMA DO PRISIONEIRO", 
                 fontsize=18, fontweight='bold', ha='left', va='top', family=fonte_padrao)
    ax_text.text(0.0, 0.90, f"Tempo (MCS): {t:05d}", 
                 fontsize=16, color='darkred', fontweight='bold', ha='left', va='top', family=fonte_padrao)
    
    # Bloco 1: Abundância Global
    texto_global = (
        "Densidade Global (Ambiente):\n"
        f"  • Vacâncias ($p$): {frac_p_global*100:05.2f}%\n"
        f"  • Ocupação Total: {(frac_c_global+frac_d_global)*100:05.3f}%\n"
    )
    ax_text.text(0.0, 0.75, texto_global, fontsize=14, ha='left', va='top', family=fonte_padrao)

    # Bloco 2: Abundância Relativa
    texto_relativo = (
        "Abundância Relativa (População):\n"
        f"  • Cooperadores ($\\rho_C$): {frac_c_rel*100:05.3f}%\n"
        f"  • Desertores ($\\rho_D$): {frac_d_rel*100:05.3f}%\n"
    )
    ax_text.text(0.0, 0.60, texto_relativo, fontsize=14, ha='left', va='top', family=fonte_padrao)

    # Bloco 3: Cinemática
    texto_cinematica = (
        "Cinemática da Colônia:\n"
        f"  • Raio de Giração ($R_g^2$): {rg2:.3f}\n"
        f"  • Anisotropia ($\\alpha$): {anisotropia:.3f}\n"
        f"  • MSD (Centro de Massa): {msd:.3f}\n"
    )
    ax_text.text(0.0, 0.45, texto_cinematica, fontsize=14, ha='left', va='top', family=fonte_padrao)

    # Bloco 4: Parâmetros Físicos
    texto_params = (
        "Parâmetros Físicos:\n"
        f"  • Tamanho ($L$): {L}$\\times${L}\n"
        f"  • Tentação ($b$): {b}\n"
        f"  • Mutação ($\\mu_C$): {TAXA_MUTACAO_C}\n"
        f"  • Mutação ($\\mu_D$): {TAXA_MUTACAO_D}\n"
    )

    ax_text.text(0.0, 0.30, texto_params, fontsize=12, ha='left', va='top', color='dimgrey', family=fonte_padrao)

    # Legendas
    legend_elements = [
        Patch(facecolor=COLORS[0], edgecolor='black', label=r'Cooperador ($C$)'),
        Patch(facecolor=COLORS[1], edgecolor='black', label=r'Desertor ($D$)'),
        Patch(facecolor=COLORS[2], edgecolor='black', label='Buraco (Vazio)')
    ]
    ax_text.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0.0, 0.0), 
                   fontsize=12, frameon=False, prop={'family': fonte_padrao, 'size': 12})
    
    # Salva a imagem em uma subpasta com o nome da semente
    pasta_semente = os.path.join(CAMINHO_FRAMES, f'semente_{semente}')
    os.makedirs(pasta_semente, exist_ok=True)
    plt.savefig(os.path.join(pasta_semente, f"frame_{t:04d}_L{L}.png"), bbox_inches='tight', dpi=120)
    plt.close()

# ==============================================================================
# 4.5 SANITIZAÇÃO DE DADOS (I/O SHIELDING)
# ==============================================================================
def sanitizar_csv_resume(caminho_csv, num_colunas_esperadas=11):
    """Verifica se a última linha do CSV está corrompida (queda de energia) e a remove."""
    with open(caminho_csv, 'r+', newline='') as f:
        linhas = f.readlines()
        # Conta os ponto e vírgulas para ver se a linha foi escrita até o fim
        if linhas and linhas[-1].count(';') + 1 != num_colunas_esperadas:
            print(f"    [!] Última linha truncada detectada em {os.path.basename(caminho_csv)} — removendo.")
            linhas = linhas[:-1]
            f.seek(0)
            f.truncate()
            f.writelines(linhas)

# ==============================================================================
# 5. O LAÇO MESTRE DE EVOLUÇÃO UNIFICADO (MODO HPC / CHECKPOINTS)
# ==============================================================================
def executar_simulacao():
    print(f"[{time.strftime('%H:%M:%S')}] Iniciando Motor Termodinâmico (Modo HPC/Checkpoint)...")
    print(f"Modo Visual (Frames): {'LIGADO' if GERAR_FRAMES else 'DESLIGADO'}")

    # ======================================================================
    # 1. PROCESSAMENTO DAS SEMENTES (ANTES DO LAÇO DOS TAMANHOS)
    # ======================================================================
    if isinstance(CONFIG_SEMENTES, int):
        np.random.seed() 
        sementes_usadas = np.random.randint(1, 9999999, size=CONFIG_SEMENTES).tolist()
    elif isinstance(CONFIG_SEMENTES, list):
        sementes_usadas = CONFIG_SEMENTES
    else:
        sementes_usadas = [42] 
        
    num_simulacoes = len(sementes_usadas)

    for L in TAMANHOS_REDE:
        pasta_L = os.path.join(CAMINHO_DADOS, f'L_{L}')
        os.makedirs(pasta_L, exist_ok=True)
        
        # Salva o log de sementes dentro da pasta específica do L
        caminho_sementes = os.path.join(pasta_L, 'sementes_usadas.txt')
        with open(caminho_sementes, 'w') as f:
            f.write(f"Tamanho do Ensemble: {num_simulacoes} rodadas\n")
            f.write(f"Sementes Utilizadas: {sementes_usadas}\n")
            
        print("\n" + "="*50)
        print(f"PROCESSANDO REDE: L = {L} ({L}x{L}) | Steps: {STEPS} | Ensemble: {num_simulacoes} rodadas")
        print("="*50)

        # --- ALOCAÇÃO DE MEMÓRIA HPC (FEITA APENAS UMA VEZ POR TAMANHO DE REDE) ---
        _labels_buf = np.zeros((L, L), dtype=np.int32)
        _queue_r_buf = np.zeros(L * L, dtype=np.int32)
        _queue_c_buf = np.zeros(L * L, dtype=np.int32)
        _offset_x_buf = np.zeros((L, L), dtype=np.int32)
        _offset_y_buf = np.zeros((L, L), dtype=np.int32)
        _percola_x_buf = np.zeros(L * L, dtype=np.bool_)
        _percola_y_buf = np.zeros(L * L, dtype=np.bool_)
        # -------------------------------

        # ======================================================================
        # 2. LOOP DO ENSEMBLE (ESCRITA DIRETA E CHECKPOINTS)
        # ======================================================================
        for idx_sim, semente in enumerate(sementes_usadas):
            print(f"\n---> Iniciando Rodada {idx_sim + 1}/{num_simulacoes} [Semente: {semente}]")
            np.random.seed(semente)
            
            caminho_csv = os.path.join(pasta_L, f'log_semente_{semente}.csv')
            caminho_chk = os.path.join(CAMINHO_CHECKPOINTS, f'chk_L{L}_semente_{semente}.npz')
            
            passo_inicial = 0
            
            # --- SISTEMA DE RESUME (CHECKPOINT BLINDADO COM RNG) ---
            if CONTINUAR_SIMULACAO and os.path.exists(caminho_chk) and os.path.exists(caminho_csv):
                print(f"    [!] Checkpoint encontrado. Restaurando estado da simulação e entropia...")
                # O allow_pickle=True é necessário para carregar o array de objetos do rng_state
                chk = np.load(caminho_chk, allow_pickle=True)
                grid = chk['grid']
                paredes_mask = chk['paredes_mask']
                tracking = chk['tracking']
                passo_inicial = int(chk['step'])
                
                # Restaura a exata sequência de números aleatórios de onde parou
                np.random.set_state(tuple(chk['rng_state']))
                
                cx_inicial, cy_inicial = tracking[0], tracking[1]
                cx_prev, cy_prev = tracking[2], tracking[3]
                cx_unwrapped, cy_unwrapped = tracking[4], tracking[5]
                msd = tracking[6]
                
                # Resgate seguro da massa (compatibilidade com checkpoints antigos)
                massa_prev = tracking[7] if len(tracking) > 7 else 0
                chk.close()

                if passo_inicial >= STEPS:
                    print(f"    ✔ Simulação já atingiu os {STEPS} passos. Pulando.")
                    continue
                    
                # --- SANITIZAÇÃO DO CSV ANTES DE CONTINUAR ---
                sanitizar_csv_resume(caminho_csv, num_colunas_esperadas=11)
                modo_abertura = 'a'
            else:
                # SÓ reseta a semente se for uma simulação virgem
                np.random.seed(semente)
                grid, paredes_mask = inicializar_universo(L)

                # --- NOVO: Trava de cinemática inicial ---
                if CALCULAR_CINEMATICA:
                    cx_inicial, cy_inicial, rg2, anisotropia, massa_prev, p_x, p_y = calcular_cinematica_completa(
                        grid == INDICE_C, L, TOPOLOGIA_TABULEIRO, _labels_buf, _queue_r_buf, _queue_c_buf, _offset_x_buf, _offset_y_buf, _percola_x_buf, _percola_y_buf
                    )
                else:
                    cx_inicial, cy_inicial, rg2, anisotropia, massa_prev, p_x, p_y = np.nan, np.nan, np.nan, np.nan, 0, False, False
                # -----------------------------------------
                cx_prev, cy_prev = cx_inicial, cy_inicial
                cx_unwrapped, cy_unwrapped = cx_inicial, cy_inicial
                msd = 0.0
                modo_abertura = 'w'
                
                if GERAR_FRAMES:
                    renderizar_frame(grid, 0, cx_inicial, cy_inicial, rg2, anisotropia, msd, L, paredes_mask, semente) 
            
            # --- ESCRITA EM TEMPO REAL NO HD ---
            buffer_dados = [] # NOVO: Cria a lista para guardar os dados na RAM
            with open(caminho_csv, modo_abertura, newline='') as f_csv:
                writer = csv.writer(f_csv, delimiter=';')
                if modo_abertura == 'w':
                    writer.writerow(['step', 'frac_c', 'frac_d', 'porosidade_p', 'rg2', 'anisotropia', 'msd', 'atividade', 'salto_identidade', 'percola_x', 'percola_y'])
                    
                for step_idx in range(passo_inicial, STEPS):
                    t = step_idx + 1
                    grid_anterior = np.copy(grid)
                    
                    # FASE 1: CENSO POPULACIONAL
                    n_c, n_d, payoffs = censo_e_payoff_numba(
                        grid, L, PAYOFF_MATRIX[0,0], PAYOFF_MATRIX[0,1], 
                        PAYOFF_MATRIX[1,0], PAYOFF_MATRIX[1,1], DIRECTIONS_ARRAY
                    )
                    
                    # FASE 2: ESCOLHA RACIONAL E EVOLUÇÃO
                    # Como não há mais mecânicas degradando a matriz dinamicamente no meio do passo,
                    # utilizamos os 'payoffs' recém calculados diretamente na regra de decisão.
                    proposed_grid_rational = escolha_racional_numba(grid, payoffs, L, K_FERMI, TAU_FERMI, DIRECTIONS_ARRAY)
                    
                    # FASE 3: MUTAÇÃO E ASSINCRONIA
                    proposed_grid_mutated = (proposed_grid_rational + 1) % NUM_STRATEGIES
                    rand_matrix = np.random.rand(L, L)
                    
                    mutate_c = (grid == ESTRATEGIA_INVASOR) & (rand_matrix < TAXA_MUTACAO_C)
                    mutate_d = (grid == ESTRATEGIA_MAR) & (rand_matrix < TAXA_MUTACAO_D)
                    mutation_mask = mutate_c | mutate_d
                    
                    proposed_grid_with_mutation = np.where(mutation_mask, proposed_grid_mutated, proposed_grid_rational)
                    
                    if PROBABILIDADE_ATUALIZACAO >= 1.0:
                        proposed_grid = proposed_grid_with_mutation
                    else:
                        celulas_ativas = np.where(grid != INDICE_BURACO)
                        num_ativos = len(celulas_ativas[0])
                        qtd_atualizar = int(round(num_ativos * PROBABILIDADE_ATUALIZACAO))
                        async_mask = np.zeros((L, L), dtype=bool)
                        
                        if qtd_atualizar > 0:
                            indices_sorteados = np.random.choice(num_ativos, qtd_atualizar, replace=False)
                            coords_r = celulas_ativas[0][indices_sorteados]
                            coords_c = celulas_ativas[1][indices_sorteados]
                            async_mask[coords_r, coords_c] = True
                        
                        proposed_grid = np.where(async_mask, proposed_grid_with_mutation, grid)
                    
                    proposed_grid[grid == INDICE_BURACO] = INDICE_BURACO
                    proposed_grid[paredes_mask] = INDICE_BURACO
                    grid = proposed_grid
                    
                    # FASE 4: EXTRAÇÃO ESTATÍSTICA E MSD
                    atividade_step = np.sum(grid != grid_anterior)
                    
                    total_uteis = (L * L) - np.sum(paredes_mask)
                    frac_c = np.sum(grid == INDICE_C) / total_uteis
                    frac_d = np.sum(grid == INDICE_D) / total_uteis
                    frac_hole = np.sum((grid == INDICE_BURACO) & (~paredes_mask)) / total_uteis
                    
                    # --- NOVO: Trava de cinemática contínua ---
                    if CALCULAR_CINEMATICA:
                        cx, cy, rg2, anisotropia, massa_atual, p_x, p_y = calcular_cinematica_completa(
                            grid == INDICE_C, L, TOPOLOGIA_TABULEIRO, _labels_buf, _queue_r_buf, _queue_c_buf, _offset_x_buf, _offset_y_buf, _percola_x_buf, _percola_y_buf, cx_prev, cy_prev
                        )
                    else:
                        cx, cy, rg2, anisotropia, massa_atual, p_x, p_y = np.nan, np.nan, np.nan, np.nan, 0, False, False
                    # ------------------------------------------
                    
                    salto_identidade = False
                    
                    if not np.isnan(cx):
                        # Telemetria: O rastreador pulou para um pedaço com menos da metade da massa?
                        if massa_prev > 0 and massa_atual < (0.5 * massa_prev):
                            salto_identidade = True
                            
                        massa_prev = massa_atual
                        
                        if np.isnan(cx_prev):
                            cx_unwrapped = cx; cy_unwrapped = cy
                            cx_inicial = cx; cy_inicial = cy
                        else:
                            dx = cx - cx_prev
                            dy = cy - cy_prev
                            
                            if TOPOLOGIA_TABULEIRO == "toroide":
                                if dx > L / 2.0: dx -= L
                                elif dx < -L / 2.0: dx += L
                                if dy > L / 2.0: dy -= L
                                elif dy < -L / 2.0: dy += L
                            
                            cx_unwrapped += dx
                            cy_unwrapped += dy
                            
                        msd = (cx_unwrapped - cx_inicial)**2 + (cy_unwrapped - cy_inicial)**2
                        
                        # A CORREÇÃO DE MEMÓRIA: Atualiza apenas se cx for válido!
                        cx_prev, cy_prev = cx, cy
                    else:
                        msd = np.nan
                        # Se cx for NaN (instabilidade circular), cx_prev e cy_prev intencionalmente 
                        # NÃO são atualizados. A memória do último local seguro é preservada!

                    # ---------------------------------------------
                    # GRAVAÇÃO CONTÍNUA (BUFFER E CHECKPOINT ATÔMICO)
                    # ---------------------------------------------
                    buffer_dados.append([t, frac_c, frac_d, frac_hole, rg2, anisotropia, msd, atividade_step, int(salto_identidade), int(p_x), int(p_y)])
                    
                    # Descarrega da memória RAM para o HD em blocos de 1000 (Evita gargalo de I/O)
                    if t % CHECKPOINT_SKIP == 0 or t == STEPS:
                        writer.writerows(buffer_dados)
                        buffer_dados.clear()
                        f_csv.flush() 
                        print(f"    Step {t:04d} | C: {frac_c*100:04.1f}% | Rg2: {rg2:.2f} | Atv: {atividade_step}")
                        
                    # Salva o "Save State" da simulação com ESCRITA ATÔMICA
                    if t % CHECKPOINT_SKIP == 0 or t == STEPS:
                        tracking_data = np.array([cx_inicial, cy_inicial, cx_prev, cy_prev, cx_unwrapped, cy_unwrapped, msd, massa_prev])
                        
                        # Correção: O sufixo temporário deve terminar em .npz para o NumPy não duplicar a extensão
                        temp_chk = caminho_chk.replace('.npz', '_temp.npz') 
                        
                        # Captura o estado exato da entropia do Numpy neste milissegundo
                        estado_rng = np.random.get_state()
                        
                        np.savez(temp_chk, grid=grid, paredes_mask=paredes_mask, tracking=tracking_data, 
                                 step=t, rng_state=np.array(estado_rng, dtype=object))
                        
                        try:
                            os.replace(temp_chk, caminho_chk)
                        except PermissionError:
                            time.sleep(0.1)
                            if os.path.exists(caminho_chk):
                                try:
                                    os.remove(caminho_chk)
                                except Exception:
                                    pass
                            os.replace(temp_chk, caminho_chk)
                        
                        print(f"    [💾 Checkpoint Salvo no passo {t}]")
                    
                    if GERAR_FRAMES and t % FRAME_SKIP == 0:
                        renderizar_frame(grid, t, cx, cy, rg2, anisotropia, msd, L, paredes_mask, semente)

        print(f"✔ Rodadas L={L} Concluídas. Dados salvos individualmente.")

    print(f"\n[{time.strftime('%H:%M:%S')}] Varredura HPC finalizada com sucesso.")

if __name__ == "__main__":
    executar_simulacao()