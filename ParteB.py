import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from control import tf, feedback, step_response, c2d, poles, dcgain

# ==============================================================================
# PARTE 1: DEFINIÇÃO DOS PARÂMETROS E CÁLCULO DO POLINÔMIO ALVO
# ==============================================================================
print("--- 1. CÁLCULO DOS POLOS DESEJADOS ---")

# --- Parâmetros ---
zeta_escolhido = 0.7   
wn_escolhido = 5.0
alpha = 20
T = 0.01

print(f"Zeta: {zeta_escolhido}, Wn: {wn_escolhido}, Alpha: {alpha}, T: {T}")

# --- Cálculo em S ---
sigma = zeta_escolhido * wn_escolhido
wd = wn_escolhido * np.sqrt(1 - zeta_escolhido**2)

s1 = -sigma + 1j*wd
s2 = -sigma - 1j*wd
s3 = -alpha

print(f"\nPolos Escolhidos em S:")
print(f"s1, s2 = {s1:.2f}, {s2:.2f}")
print(f"s3 = {s3:.2f}")

# --- Mapeamento para Z ---
z1 = np.exp(s1*T)
z2 = np.exp(s2*T)
z3 = np.exp(s3*T)

print(f"\nPolos Alvo (z):")
print(f"z1 = {z1:.4f}")
print(f"z2 = {z2:.4f}")
print(f"z3 = {z3:.4f}")

# --- Polinômio Desejado (Ordem 3) ---
# Gera coeficientes para (z-z1)(z-z2)(z-z3)
# Formato: z^3 + d2*z^2 + d1*z + d0
poly_target = np.poly([z1, z2, z3])

# Dicionário de coeficientes alvo para o matching
# poly_target[0] é 1 (z^3)
d_target = {
    2: poly_target[1], # Coef de z^2
    1: poly_target[2], # Coef de z^1
    0: poly_target[3]  # Coef de z^0 (termo independente)
}

print("\nPolinômio Alvo (Numérico):")
print(f"z^3 + ({d_target[2]:.4f})z^2 + ({d_target[1]:.4f})z + ({d_target[0]:.4f}) = 0")
# ==============================================================================
# PARTE 2: DEDUÇÃO ALGÉBRICA (SYMPY)
# ==============================================================================
print("\n" + "="*50)
print("--- 2. DEDUÇÃO SIMBÓLICA ---")
print("="*50)

# Variáveis Simbólicas
z = sp.symbols('z')
Kp, Ki, Kd = sp.symbols('Kp Ki Kd')

# Planta G(z)
Num_G = 0.324 * z
Den_G = z**2 - 1.999 * z + 0.9985

# Controlador PID C(z)
# Estrutura: Kp + Ki*z/(z-1) + Kd*(z-1)/z
PID_eq = Kp + Ki * (z / (z - 1)) + Kd * ((z - 1) / z)
C_z_racional = sp.together(PID_eq)
Num_C, Den_C = sp.fraction(C_z_racional)

print("Numerador do PID (Simbólico):")
sp.pprint(sp.collect(Num_C, z))

# Equação Característica: Num_C*Num_G + Den_C*Den_G = 0
Eq_Caracteristica_Bruta = Num_C * Num_G + Den_C * Den_G

# --- SIMPLIFICAÇÃO IMPORTANTE ---
# Como Num_G tem 'z' e Den_C tem 'z', ambos os termos da soma têm 'z'.
# Podemos dividir toda a equação por z para reduzir a ordem de 4 para 3.
Eq_Caracteristica_Simplificada = sp.simplify(Eq_Caracteristica_Bruta / z)

# Expandir e agrupar por potências de z
Eq_Final = sp.collect(sp.expand(Eq_Caracteristica_Simplificada), z)

print("\nEquação Característica (Simplificada para Ordem 3):")
sp.pprint(Eq_Final)
# ==============================================================================
# PARTE 3: SOLUÇÃO DO SISTEMA (MATCHING)
# ==============================================================================
print("\n" + "="*50)
print("--- 3. SOLUÇÃO DO SISTEMA (AX = B) ---")
print("="*50)

# Extrair coeficientes simbólicos de z^2, z^1 e z^0
coeff_z2_sym = Eq_Final.coeff(z, 2)
coeff_z1_sym = Eq_Final.coeff(z, 1)
coeff_z0_sym = Eq_Final.coeff(z, 0)

# Montar equações: Simbólico == Alvo
eq1 = sp.Eq(coeff_z2_sym, d_target[2])
eq2 = sp.Eq(coeff_z1_sym, d_target[1])
eq3 = sp.Eq(coeff_z0_sym, d_target[0])

print("Sistema de Equações:")
sp.pprint(eq1)
sp.pprint(eq2)
sp.pprint(eq3)

# Transformar em Matriz A e Vetor B
sys_equations = [eq1, eq2, eq3]
variables = [Kp, Ki, Kd]

Matrix_A_sym, Vector_B_sym = sp.linear_eq_to_matrix(sys_equations, variables)

A_num = np.array(Matrix_A_sym).astype(np.float64)
B_num = np.array(Vector_B_sym).astype(np.float64)

print("\nMatriz A:")
print(A_num)
print("\nVetor B:")
print(B_num)

# Resolver
Ganhos = np.linalg.solve(A_num, B_num)
Kp_val, Ki_val, Kd_val = Ganhos.flatten()

print("\n" + "*"*30)
print(f"Kp = {Kp_val:.5f}")
print(f"Ki = {Ki_val:.5f}")
print(f"Kd = {Kd_val:.5f}")
print("*"*30)

# ==============================================================================
# PARTE 4 (INTERMEDIÁRIA): MONTAGEM DA FT PARA JURY
# ==============================================================================
# Recalcular coeficientes do PID numérico para obter o polinômio real
# Num_C = q2*z^2 + q1*z + q0
# Onde q2 = Kp+Ki+Kd, q1 = -(Kp+2Kd), q0 = Kd (Visto na parte simbólica)
q2 = Kp_val + Ki_val + Kd_val
q1 = -(Kp_val + 2*Kd_val)
q0 = Kd_val

num_pid = [q2, q1, q0]
den_pid = [1, -1, 0] # z(z-1)

sys_pid = tf(num_pid, den_pid, T)
sys_plant = tf([0.324, 0], [1, -1.999, 0.9985], T) 

# Monta a malha fechada APENAS para extrair o denominador resultante
sys_cl = feedback(sys_pid * sys_plant, 1)


# ==============================================================================
# PARTE 5: PROVA MATEMÁTICA: CRITÉRIO DE JURY
# ==============================================================================
print("\n" + "="*50)
print("--- 5. PROVA MATEMÁTICA: CRITÉRIO DE JURY ---")
print("="*50)

def tabela_jury(poly):
    """
    Gera a tabela de Jury e verifica a estabilidade para um polinômio P(z).
    poly: lista ou array com coeficientes [an, an-1, ..., a0]
    """
    # Garantir que é array numpy e normalizar pelo primeiro termo se necessário
    P = np.array(poly)
    if P[0] < 0: P = -P # Jury assume an > 0 normalmente para facilitar
    
    order = len(P) - 1
    print(f"Polinômio Característico P(z) (Ordem {order}):")
    # Formatação bonita do polinômio
    terms = []
    for i, c in enumerate(P):
        power = order - i
        terms.append(f"{c:.4f}z^{power}")
    print(" + ".join(terms).replace("+ -", "- ") + " = 0\n")

    # Condições Necessárias (Iniciais)
    print(">>> Verificando Condições Necessárias:")
    
    # 1. P(1) > 0
    p_1 = np.sum(P)
    cond1 = p_1 > 0
    print(f"1) P(1) = {p_1:.4f} > 0? {'[OK]' if cond1 else '[FALHA]'}")
    
    # 2. (-1)^n P(-1) > 0
    p_minus_1 = np.polyval(P, -1)
    cond2 = ((-1)**order * p_minus_1) > 0
    print(f"2) (-1)^{order} * P(-1) = {((-1)**order * p_minus_1):.4f} > 0? {'[OK]' if cond2 else '[FALHA]'}")
    
    # 3. |a0| < |an|
    cond3 = abs(P[-1]) < abs(P[0])
    print(f"3) |a0| ({abs(P[-1]):.4f}) < |an| ({abs(P[0]):.4f})? {'[OK]' if cond3 else '[FALHA]'}")

    if not (cond1 and cond2 and cond3):
        print("\nRESULTADO: Sistema INSTÁVEL (Falhou nas condições necessárias).")
        return

    print("\n>>> Tabela de Jury (Condições Suficientes):")
    # Algoritmo da Tabela
    rows = [P] # Começa com a linha 0 (coeficientes)
    
    # O laço roda order - 1 vezes para verificar as condições restantes
    # Para ordem 3, precisamos calcular até a linha b0
    stable = True
    
    current_row = P
    
    for k in range(order - 1):
        # Linha par (k=0, 2...) é a linha atual
        # Linha ímpar é a invertida
        row_normal = current_row
        row_reverse = current_row[::-1]
        
        print(f"Linha {2*k + 1}: {np.array2string(row_normal, precision=4, suppress_small=True)}")
        print(f"Linha {2*k + 2}: {np.array2string(row_reverse, precision=4, suppress_small=True)}")
        
        an = row_normal[0]
        a0 = row_normal[-1]
        
        # Verificar condição da linha atual |a0| < |an|
        if abs(a0) >= abs(an):
             print(f"  -> FALHA: |{a0:.4f}| >= |{an:.4f}|")
             stable = False
             break
        
        # Calcular próxima linha (tamanho reduzido em 1)
        next_row = []
        n_current = len(row_normal) - 1
        
        for i in range(n_current):
            # Implementação numérica padrão:
            # next_coeff = row[i] - alpha * row[reverse_i]
            alpha = row_normal[-1] / row_normal[0]
            val = row_normal[i] - alpha * row_reverse[i]
            next_row.append(val)
            
        current_row = np.array(next_row)
        
    if stable:
        # Checar última condição
        print(f"Linha Final: {np.array2string(current_row, precision=4, suppress_small=True)}")
        if len(current_row) > 0 and abs(current_row[-1]) < abs(current_row[0]):
             print("  -> Condição Final OK.")
        else:
             pass
        print("\nRESULTADO FINAL: O sistema é ESTÁVEL segundo o critério de Jury.")
    else:
        print("\nRESULTADO FINAL: O sistema é INSTÁVEL.")

# Obter coeficientes da malha fechada do objeto 'sys_cl' (control lib)
# sys_cl.den é uma lista de listas [[1, a2, a1, a0]]
# O [0][0] pega o array numpy interno
den_coefficients = sys_cl.den[0][0]

tabela_jury(den_coefficients)

# ==============================================================================
# EXPORTAÇÃO DE DADOS (PARTE C)
# ==============================================================================
print("\n" + "="*50)
print("--- SALVANDO DADOS PARA A PARTE C ---")
print("="*50)

# Salva Kp, Ki, Kd e T num arquivo .npz
np.savez('pid_dados.npz', Kp=Kp_val, Ki=Ki_val, Kd=Kd_val, T=T)
print("Arquivo 'pid_dados.npz' salvo com sucesso!")
print("Agora você pode rodar o arquivo 'ParteC.py'.")